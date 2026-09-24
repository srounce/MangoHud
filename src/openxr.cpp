/* OpenXR API layer: measures frame timing at xrEndFrame instead of at the
 * mirror window's present and composites the HUD into the headset view as a
 * head-locked quad layer. Only Vulkan sessions are drawn; other sessions pass
 * through untouched. */

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <spdlog/spdlog.h>

#include <vulkan/vulkan.h>
#define XR_NO_PROTOTYPES
#define XR_USE_GRAPHICS_API_VULKAN
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>

#include "mesa/util/macros.h"
#include "overlay.h"
#include "blacklist.h"
#include "vulkan_xr.h"

extern "C" PUBLIC XRAPI_ATTR XrResult XRAPI_CALL
xrNegotiateLoaderApiLayerInterface(const XrNegotiateLoaderInfo *loaderInfo,
                                   const char *layerName,
                                   XrNegotiateApiLayerRequest *apiLayerRequest);

namespace {

const char layer_name_prefix[] = "XR_APILAYER_MANGOHUD_overlay";

struct xr_instance_data {
   XrInstance instance = XR_NULL_HANDLE;
   PFN_xrGetInstanceProcAddr next_gipa = nullptr;

   PFN_xrDestroyInstance DestroyInstance = nullptr;
   PFN_xrCreateSession CreateSession = nullptr;
   PFN_xrDestroySession DestroySession = nullptr;
   PFN_xrEndFrame EndFrame = nullptr;
   PFN_xrCreateSwapchain CreateSwapchain = nullptr;
   PFN_xrDestroySwapchain DestroySwapchain = nullptr;
   PFN_xrEnumerateSwapchainFormats EnumerateSwapchainFormats = nullptr;
   PFN_xrEnumerateSwapchainImages EnumerateSwapchainImages = nullptr;
   PFN_xrAcquireSwapchainImage AcquireSwapchainImage = nullptr;
   PFN_xrWaitSwapchainImage WaitSwapchainImage = nullptr;
   PFN_xrReleaseSwapchainImage ReleaseSwapchainImage = nullptr;
   PFN_xrCreateReferenceSpace CreateReferenceSpace = nullptr;
   PFN_xrDestroySpace DestroySpace = nullptr;
};

struct xr_session_data {
   xr_instance_data *instance = nullptr;
   XrSession session = XR_NULL_HANDLE;
   bool vulkan = false;
   XrGraphicsBindingVulkanKHR binding {};

   bool init_failed = false;
   XrSwapchain swapchain = XR_NULL_HANDLE;
   XrSpace view_space = XR_NULL_HANDLE;
   VkFormat format = VK_FORMAT_UNDEFINED;
   uint32_t width = 0, height = 0;
   xr_vk_target *target = nullptr;
   bool announced = false;
   bool defer_logged = false;
};

std::mutex xr_lock;
std::unordered_map<XrInstance, xr_instance_data *> xr_instances;
std::unordered_map<XrSession, xr_session_data *> xr_sessions;

xr_instance_data *find_instance(XrInstance instance)
{
   std::lock_guard<std::mutex> lk(xr_lock);
   auto it = xr_instances.find(instance);
   return it == xr_instances.end() ? nullptr : it->second;
}

xr_session_data *find_session(XrSession session)
{
   std::lock_guard<std::mutex> lk(xr_lock);
   auto it = xr_sessions.find(session);
   return it == xr_sessions.end() ? nullptr : it->second;
}

bool xr_ok(XrResult result, const char *what)
{
   if (XR_SUCCEEDED(result))
      return true;
   SPDLOG_ERROR("{} failed with XrResult {}", what, (int)result);
   return false;
}

template<typename T>
void load_next(xr_instance_data *data, const char *name, T& fn)
{
   PFN_xrVoidFunction f = nullptr;
   if (XR_FAILED(data->next_gipa(data->instance, name, &f)))
      f = nullptr;
   fn = reinterpret_cast<T>(f);
   if (!fn)
      SPDLOG_ERROR("OpenXR runtime does not provide {}", name);
}

void destroy_session_overlay(xr_session_data *sd)
{
   xr_instance_data *inst = sd->instance;
   if (sd->target) {
      xr_vk_target_destroy(sd->target);
      sd->target = nullptr;
   }
   if (sd->swapchain != XR_NULL_HANDLE) {
      inst->DestroySwapchain(sd->swapchain);
      sd->swapchain = XR_NULL_HANDLE;
   }
   if (sd->view_space != XR_NULL_HANDLE) {
      inst->DestroySpace(sd->view_space);
      sd->view_space = XR_NULL_HANDLE;
   }
}

/* The OpenXR layer and the Vulkan overlay may be different copies of
 * libMangoHud.so (e.g. a distro Vulkan layer beside a locally built OpenXR
 * layer), each with its own globals. The Vulkan layer parses the config and
 * sets up stats in its copy; if that has not happened in ours, do it here so
 * the OpenXR layer is self-sufficient. Runs once per process. */
void ensure_mangohud_config()
{
   static std::once_flag once;
   std::call_once(once, [] {
      if (get_params_nonblocking())
         return; /* the Vulkan overlay in this same library already did it */
      SPDLOG_DEBUG("OpenXR: no Vulkan overlay in this library, initialising MangoHud here");
      static overlay_params params;
      parse_overlay_config(&params, getenv("MANGOHUD_CONFIG"), false);
      init_system_info();
      init_cpu_stats(params);
   });
}

bool init_session_overlay(xr_session_data *sd)
{
   xr_instance_data *inst = sd->instance;

   /* Do not block the app's frame thread: if the Vulkan layer has not parsed
    * the config yet, skip this frame and try again on the next one. */
   auto params = get_params_nonblocking();
   if (!params) {
      SPDLOG_DEBUG("OpenXR: config not ready yet, deferring HUD setup a frame");
      return false;
   }

   SPDLOG_DEBUG("OpenXR: enumerating swapchain formats");
   uint32_t n_formats = 0;
   if (!xr_ok(inst->EnumerateSwapchainFormats(sd->session, 0, &n_formats, nullptr), "xrEnumerateSwapchainFormats"))
      return false;
   std::vector<int64_t> formats(n_formats);
   if (!xr_ok(inst->EnumerateSwapchainFormats(sd->session, n_formats, &n_formats, formats.data()), "xrEnumerateSwapchainFormats"))
      return false;

   /* sRGB first: the compositor blends in linear light and the HUD colors are
    * sRGB encoded, which the Vulkan side accounts for per format. */
   static const VkFormat preferred[] = {
      VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB,
      VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM,
   };
   sd->format = VK_FORMAT_UNDEFINED;
   for (VkFormat f : preferred) {
      if (std::find(formats.begin(), formats.end(), (int64_t)f) != formats.end()) {
         sd->format = f;
         break;
      }
   }
   if (sd->format == VK_FORMAT_UNDEFINED) {
      SPDLOG_ERROR("OpenXR runtime offers no 8-bit RGBA swapchain format for the HUD");
      return false;
   }

   sd->width = sd->height = std::max(params->vr_resolution, 64u);
   XrSwapchainCreateInfo swapchain_info { XR_TYPE_SWAPCHAIN_CREATE_INFO };
   swapchain_info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT;
   swapchain_info.format = sd->format;
   swapchain_info.sampleCount = 1;
   swapchain_info.width = sd->width;
   swapchain_info.height = sd->height;
   swapchain_info.faceCount = 1;
   swapchain_info.arraySize = 1;
   swapchain_info.mipCount = 1;
   if (!xr_ok(inst->CreateSwapchain(sd->session, &swapchain_info, &sd->swapchain), "xrCreateSwapchain"))
      return false;

   uint32_t n_images = 0;
   if (!xr_ok(inst->EnumerateSwapchainImages(sd->swapchain, 0, &n_images, nullptr), "xrEnumerateSwapchainImages"))
      return false;
   std::vector<XrSwapchainImageVulkanKHR> images(n_images, { XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR, nullptr, VK_NULL_HANDLE });
   if (!xr_ok(inst->EnumerateSwapchainImages(sd->swapchain, n_images, &n_images,
                                             reinterpret_cast<XrSwapchainImageBaseHeader *>(images.data())),
              "xrEnumerateSwapchainImages"))
      return false;

   XrReferenceSpaceCreateInfo space_info { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
   space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
   space_info.poseInReferenceSpace.orientation = { 0.f, 0.f, 0.f, 1.f };
   space_info.poseInReferenceSpace.position = { 0.f, 0.f, 0.f };
   if (!xr_ok(inst->CreateReferenceSpace(sd->session, &space_info, &sd->view_space), "xrCreateReferenceSpace"))
      return false;

   xr_vk_target_info target_info {};
   target_info.instance = sd->binding.instance;
   target_info.device = sd->binding.device;
   target_info.queue_family_index = sd->binding.queueFamilyIndex;
   target_info.queue_index = sd->binding.queueIndex;
   target_info.format = sd->format;
   target_info.width = sd->width;
   target_info.height = sd->height;
   for (auto& image : images)
      target_info.images.push_back(image.image);

   SPDLOG_DEBUG("OpenXR HUD swapchain ready: {}x{} format {} images {}, creating Vulkan render target",
               sd->width, sd->height, (int)sd->format, n_images);
   sd->target = xr_vk_target_create(target_info);
   if (!sd->target)
      return false;

   SPDLOG_DEBUG("OpenXR headset HUD initialised");
   return true;
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrEndFrame(XrSession session, const XrFrameEndInfo *frameEndInfo)
{
   xr_session_data *sd = find_session(session);
   if (!sd)
      return XR_ERROR_HANDLE_INVALID;
   xr_instance_data *inst = sd->instance;

   if (!sd->vulkan || sd->init_failed || !frameEndInfo)
      return inst->EndFrame(session, frameEndInfo);

   if (!sd->target) {
      /* Make sure the config and stats exist in this library, then set up. */
      ensure_mangohud_config();
      if (!get_params_nonblocking()) {
         if (!sd->defer_logged) {
            SPDLOG_WARN("OpenXR: MangoHud config unavailable even after init; deferring HUD");
            sd->defer_logged = true;
         }
         return inst->EndFrame(session, frameEndInfo);
      }

      SPDLOG_DEBUG("First xrEndFrame for this session, setting up the headset HUD");
      if (!init_session_overlay(sd)) {
         SPDLOG_ERROR("Giving up on drawing the HUD in the headset for this session");
         destroy_session_overlay(sd);
         sd->init_failed = true;
         return inst->EndFrame(session, frameEndInfo);
      }
   }

   if (!xr_vk_target_update(sd->target))
      return inst->EndFrame(session, frameEndInfo);

   uint32_t image_index = 0;
   XrSwapchainImageAcquireInfo acquire_info { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
   if (!xr_ok(inst->AcquireSwapchainImage(sd->swapchain, &acquire_info, &image_index), "xrAcquireSwapchainImage"))
      return inst->EndFrame(session, frameEndInfo);

   XrSwapchainImageWaitInfo wait_info { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
   wait_info.timeout = XR_INFINITE_DURATION;
   bool drawn = xr_ok(inst->WaitSwapchainImage(sd->swapchain, &wait_info), "xrWaitSwapchainImage");
   if (drawn)
      xr_vk_target_draw(sd->target, image_index);

   /* Release balances the acquire even on wait failure, or the image stays
    * acquired forever. */
   XrSwapchainImageReleaseInfo release_info { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
   xr_ok(inst->ReleaseSwapchainImage(sd->swapchain, &release_info), "xrReleaseSwapchainImage");

   /* Without a wait the image holds no HUD, so submit the runtime's own layers
    * untouched rather than a quad over garbage. */
   if (!drawn)
      return inst->EndFrame(session, frameEndInfo);

   auto params = get_params_nonblocking();
   if (!params)
      return inst->EndFrame(session, frameEndInfo);
   XrCompositionLayerQuad quad { XR_TYPE_COMPOSITION_LAYER_QUAD };
   quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
   quad.space = sd->view_space;
   quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
   quad.subImage.swapchain = sd->swapchain;
   quad.subImage.imageRect.offset = { 0, 0 };
   quad.subImage.imageRect.extent = { (int32_t)sd->width, (int32_t)sd->height };
   quad.subImage.imageArrayIndex = 0;
   quad.pose.orientation = { 0.f, 0.f, 0.f, 1.f };
   quad.pose.position = { params->vr_offset_x, params->vr_offset_y, -params->vr_distance };
   quad.size = { params->vr_size, params->vr_size * (float)sd->height / (float)sd->width };

   std::vector<const XrCompositionLayerBaseHeader *> layers(frameEndInfo->layers,
                                                            frameEndInfo->layers + frameEndInfo->layerCount);
   layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader *>(&quad));
   if (!sd->announced) {
      SPDLOG_DEBUG("OpenXR HUD quad submitted: {:.2f}x{:.2f} m at {:.2f} m in view space",
                   quad.size.width, quad.size.height, params->vr_distance);
      sd->announced = true;
   }

   XrFrameEndInfo end_info = *frameEndInfo;
   end_info.layerCount = (uint32_t)layers.size();
   end_info.layers = layers.data();
   return inst->EndFrame(session, &end_info);
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrCreateSession(XrInstance instance, const XrSessionCreateInfo *createInfo, XrSession *session)
{
   xr_instance_data *inst = find_instance(instance);
   if (!inst)
      return XR_ERROR_HANDLE_INVALID;

   XrResult result = inst->CreateSession(instance, createInfo, session);
   if (XR_FAILED(result))
      return result;

   auto *sd = new xr_session_data();
   sd->instance = inst;
   sd->session = *session;
   for (auto *s = static_cast<const XrBaseInStructure *>(createInfo->next); s; s = s->next) {
      if (s->type == XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR) {
         sd->binding = *reinterpret_cast<const XrGraphicsBindingVulkanKHR *>(s);
         sd->vulkan = true;
      }
   }
   if (sd->vulkan)
      SPDLOG_DEBUG("OpenXR Vulkan session created: VkDevice {} queue family {} index {}",
                  (void *)sd->binding.device, sd->binding.queueFamilyIndex, sd->binding.queueIndex);
   else
      SPDLOG_INFO("OpenXR session is not Vulkan, the HUD will not be drawn in the headset");

   std::lock_guard<std::mutex> lk(xr_lock);
   xr_sessions[*session] = sd;
   return result;
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrDestroySession(XrSession session)
{
   xr_session_data *sd = find_session(session);
   if (!sd)
      return XR_ERROR_HANDLE_INVALID;

   destroy_session_overlay(sd);
   XrResult result = sd->instance->DestroySession(session);

   std::lock_guard<std::mutex> lk(xr_lock);
   xr_sessions.erase(session);
   delete sd;
   return result;
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrDestroyInstance(XrInstance instance)
{
   xr_instance_data *inst = find_instance(instance);
   if (!inst)
      return XR_ERROR_HANDLE_INVALID;

   /* Sessions the application never destroyed go away with the instance. */
   std::vector<xr_session_data *> orphans;
   {
      std::lock_guard<std::mutex> lk(xr_lock);
      for (auto it = xr_sessions.begin(); it != xr_sessions.end();) {
         if (it->second->instance == inst) {
            orphans.push_back(it->second);
            it = xr_sessions.erase(it);
         } else {
            ++it;
         }
      }
   }
   for (auto *sd : orphans) {
      destroy_session_overlay(sd);
      delete sd;
   }

   XrResult result = inst->DestroyInstance(instance);

   std::lock_guard<std::mutex> lk(xr_lock);
   xr_instances.erase(instance);
   delete inst;
   return result;
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrGetInstanceProcAddr(XrInstance instance, const char *name, PFN_xrVoidFunction *function);

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrCreateApiLayerInstance(const XrInstanceCreateInfo *info,
                                                                const XrApiLayerCreateInfo *layerInfo,
                                                                XrInstance *instance)
{
   if (!layerInfo || !info || !instance ||
       layerInfo->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_CREATE_INFO ||
       layerInfo->structVersion != XR_API_LAYER_CREATE_INFO_STRUCT_VERSION ||
       layerInfo->structSize != sizeof(XrApiLayerCreateInfo) ||
       !layerInfo->nextInfo ||
       layerInfo->nextInfo->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_NEXT_INFO ||
       layerInfo->nextInfo->structVersion != XR_API_LAYER_NEXT_INFO_STRUCT_VERSION ||
       layerInfo->nextInfo->structSize != sizeof(XrApiLayerNextInfo) ||
       strncmp(layerInfo->nextInfo->layerName, layer_name_prefix, sizeof(layer_name_prefix) - 1) != 0 ||
       !layerInfo->nextInfo->nextGetInstanceProcAddr ||
       !layerInfo->nextInfo->nextCreateApiLayerInstance) {
      SPDLOG_ERROR("OpenXR loader handed the layer an unexpected XrApiLayerCreateInfo");
      return XR_ERROR_INITIALIZATION_FAILED;
   }

   XrApiLayerCreateInfo next_info = *layerInfo;
   next_info.nextInfo = layerInfo->nextInfo->next;
   XrResult result = layerInfo->nextInfo->nextCreateApiLayerInstance(info, &next_info, instance);
   if (XR_FAILED(result))
      return result;

   auto *inst = new xr_instance_data();
   inst->instance = *instance;
   inst->next_gipa = layerInfo->nextInfo->nextGetInstanceProcAddr;
   load_next(inst, "xrDestroyInstance", inst->DestroyInstance);
   load_next(inst, "xrCreateSession", inst->CreateSession);
   load_next(inst, "xrDestroySession", inst->DestroySession);
   load_next(inst, "xrEndFrame", inst->EndFrame);
   load_next(inst, "xrCreateSwapchain", inst->CreateSwapchain);
   load_next(inst, "xrDestroySwapchain", inst->DestroySwapchain);
   load_next(inst, "xrEnumerateSwapchainFormats", inst->EnumerateSwapchainFormats);
   load_next(inst, "xrEnumerateSwapchainImages", inst->EnumerateSwapchainImages);
   load_next(inst, "xrAcquireSwapchainImage", inst->AcquireSwapchainImage);
   load_next(inst, "xrWaitSwapchainImage", inst->WaitSwapchainImage);
   load_next(inst, "xrReleaseSwapchainImage", inst->ReleaseSwapchainImage);
   load_next(inst, "xrCreateReferenceSpace", inst->CreateReferenceSpace);
   load_next(inst, "xrDestroySpace", inst->DestroySpace);

   std::lock_guard<std::mutex> lk(xr_lock);
   xr_instances[*instance] = inst;
   return result;
}

XRAPI_ATTR XrResult XRAPI_CALL overlay_xrGetInstanceProcAddr(XrInstance instance, const char *name, PFN_xrVoidFunction *function)
{
   static const struct {
      const char *name;
      PFN_xrVoidFunction fn;
      bool always;
   } hooks[] = {
      { "xrGetInstanceProcAddr", reinterpret_cast<PFN_xrVoidFunction>(overlay_xrGetInstanceProcAddr), true },
      { "xrDestroyInstance", reinterpret_cast<PFN_xrVoidFunction>(overlay_xrDestroyInstance), true },
      { "xrCreateSession", reinterpret_cast<PFN_xrVoidFunction>(overlay_xrCreateSession), false },
      { "xrDestroySession", reinterpret_cast<PFN_xrVoidFunction>(overlay_xrDestroySession), false },
      { "xrEndFrame", reinterpret_cast<PFN_xrVoidFunction>(overlay_xrEndFrame), false },
   };

   if (!name || !function)
      return XR_ERROR_VALIDATION_FAILURE;
   *function = nullptr;

   for (const auto& hook : hooks) {
      if ((hook.always || !is_blacklisted()) && strcmp(name, hook.name) == 0) {
         *function = hook.fn;
         return XR_SUCCESS;
      }
   }

   xr_instance_data *inst = find_instance(instance);
   if (!inst)
      return XR_ERROR_HANDLE_INVALID;
   return inst->next_gipa(instance, name, function);
}

} // namespace

extern "C" PUBLIC XRAPI_ATTR XrResult XRAPI_CALL
xrNegotiateLoaderApiLayerInterface(const XrNegotiateLoaderInfo *loaderInfo,
                                   const char *layerName,
                                   XrNegotiateApiLayerRequest *apiLayerRequest)
{
   init_spdlog();

   if (!loaderInfo || !apiLayerRequest ||
       loaderInfo->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO ||
       loaderInfo->structVersion != XR_LOADER_INFO_STRUCT_VERSION ||
       loaderInfo->structSize != sizeof(XrNegotiateLoaderInfo) ||
       apiLayerRequest->structType != XR_LOADER_INTERFACE_STRUCT_API_LAYER_REQUEST ||
       apiLayerRequest->structVersion != XR_API_LAYER_INFO_STRUCT_VERSION ||
       apiLayerRequest->structSize != sizeof(XrNegotiateApiLayerRequest) ||
       loaderInfo->minInterfaceVersion > XR_CURRENT_LOADER_API_LAYER_VERSION ||
       loaderInfo->maxInterfaceVersion < XR_CURRENT_LOADER_API_LAYER_VERSION ||
       loaderInfo->minApiVersion > XR_CURRENT_API_VERSION ||
       loaderInfo->maxApiVersion < XR_CURRENT_API_VERSION) {
      SPDLOG_ERROR("OpenXR loader interface negotiation failed for {}", layerName ? layerName : "?");
      return XR_ERROR_INITIALIZATION_FAILED;
   }

   apiLayerRequest->layerInterfaceVersion = XR_CURRENT_LOADER_API_LAYER_VERSION;
   apiLayerRequest->layerApiVersion = XR_CURRENT_API_VERSION;
   apiLayerRequest->getInstanceProcAddr = overlay_xrGetInstanceProcAddr;
   apiLayerRequest->createApiLayerInstance = overlay_xrCreateApiLayerInstance;
   SPDLOG_DEBUG("MangoHud OpenXR layer loaded, negotiation succeeded");
   return XR_SUCCESS;
}
