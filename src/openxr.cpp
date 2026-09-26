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
#include "fps_limiter.h"
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
   XrSpace space = XR_NULL_HANDLE;
   VkFormat format = VK_FORMAT_UNDEFINED;
   uint32_t width = 0, height = 0;
   xr_vk_target *target = nullptr;
   bool announced = false;
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

/* The OpenXR loader only checks that the enable variable is set, while the
 * Vulkan manifest matches its value ("1"). Match the value here too, so
 * MANGOHUD=0 disables both layers instead of just the Vulkan one. */
bool mangohud_enabled()
{
   static const bool enabled = [] {
      const char *e = getenv("MANGOHUD");
      return e && strcmp(e, "1") == 0;
   }();
   return enabled;
}

/* fps_limit is normally applied around the mirror present, which yields the
 * limiter while the headset loop runs; apply it here instead. */
XrResult end_frame_limited(xr_instance_data *inst, XrSession session, const XrFrameEndInfo *info)
{
   if (fps_limiter)
      fps_limiter->limit(true);
   XrResult result = inst->EndFrame(session, info);
   if (fps_limiter)
      fps_limiter->limit(false);
   return result;
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
   if (sd->space != XR_NULL_HANDLE) {
      inst->DestroySpace(sd->space);
      sd->space = XR_NULL_HANDLE;
   }
}

bool init_session_overlay(xr_session_data *sd, const overlay_params& params)
{
   xr_instance_data *inst = sd->instance;

   uint32_t n_formats = 0;
   if (!xr_ok(inst->EnumerateSwapchainFormats(sd->session, 0, &n_formats, nullptr), "xrEnumerateSwapchainFormats"))
      return false;
   std::vector<int64_t> formats(n_formats);
   if (!xr_ok(inst->EnumerateSwapchainFormats(sd->session, n_formats, &n_formats, formats.data()), "xrEnumerateSwapchainFormats"))
      return false;

   /* sRGB only: the compositor treats any non-sRGB format as linear, which
    * would composite the sRGB-encoded HUD colors washed out. The Vulkan side
    * linearises the colors for these formats. */
   static const VkFormat preferred[] = {
      VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB,
   };
   sd->format = VK_FORMAT_UNDEFINED;
   for (VkFormat f : preferred) {
      if (std::find(formats.begin(), formats.end(), (int64_t)f) != formats.end()) {
         sd->format = f;
         break;
      }
   }
   if (sd->format == VK_FORMAT_UNDEFINED) {
      SPDLOG_ERROR("OpenXR runtime offers no sRGB RGBA8 swapchain format for the HUD");
      return false;
   }

   /* HUDs are tall and narrow, so give the canvas vertical headroom; the quad
    * is cropped to the content each frame, so the extra height is free. */
   uint32_t res = std::max(params.vr_resolution, 64u);
   sd->width = res;
   sd->height = res * 2;
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

   /* Anchor: "view" head-locks the HUD to the eyes (it follows the gaze);
    * "local" and "stage" pin it in world space relative to the seated or
    * play-space origin, so it stays put and the user can look around it. */
   XrReferenceSpaceType ref_type = XR_REFERENCE_SPACE_TYPE_VIEW;
   if (params.vr_anchor == "local")
      ref_type = XR_REFERENCE_SPACE_TYPE_LOCAL;
   else if (params.vr_anchor == "stage")
      ref_type = XR_REFERENCE_SPACE_TYPE_STAGE;
   else if (params.vr_anchor != "view")
      SPDLOG_WARN("Unknown vr_anchor '{}', head-locking the HUD", params.vr_anchor);

   XrReferenceSpaceCreateInfo space_info { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
   space_info.referenceSpaceType = ref_type;
   space_info.poseInReferenceSpace.orientation = { 0.f, 0.f, 0.f, 1.f };
   space_info.poseInReferenceSpace.position = { 0.f, 0.f, 0.f };
   XrResult space_result = inst->CreateReferenceSpace(sd->session, &space_info, &sd->space);
   if (XR_FAILED(space_result) && ref_type == XR_REFERENCE_SPACE_TYPE_STAGE) {
      /* Not every runtime or setup has a configured play space. */
      SPDLOG_WARN("vr_anchor=stage unavailable, falling back to local");
      space_info.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
      space_result = inst->CreateReferenceSpace(sd->session, &space_info, &sd->space);
   }
   if (!xr_ok(space_result, "xrCreateReferenceSpace"))
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

   SPDLOG_INFO("OpenXR headset HUD initialised");
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
      /* The Vulkan layer in this library parsed the config when it created the
       * device this session is bound to (checked at xrCreateSession), so this
       * is only a guard. */
      auto params = get_params_nonblocking();
      if (!params)
         return inst->EndFrame(session, frameEndInfo);

      SPDLOG_DEBUG("First xrEndFrame for this session, setting up the headset HUD");
      if (!init_session_overlay(sd, *params)) {
         SPDLOG_ERROR("Giving up on drawing the HUD in the headset for this session");
         destroy_session_overlay(sd);
         sd->init_failed = true;
         return inst->EndFrame(session, frameEndInfo);
      }
   }

   if (!xr_vk_target_update(sd->target))
      return end_frame_limited(inst, session, frameEndInfo);

   uint32_t image_index = 0;
   XrSwapchainImageAcquireInfo acquire_info { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
   if (!xr_ok(inst->AcquireSwapchainImage(sd->swapchain, &acquire_info, &image_index), "xrAcquireSwapchainImage"))
      return end_frame_limited(inst, session, frameEndInfo);

   XrSwapchainImageWaitInfo wait_info { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
   wait_info.timeout = XR_INFINITE_DURATION;
   /* Only true when the HUD was actually recorded into the image; the draw can
    * still decline, e.g. no_display flipped by a config reload mid-frame. */
   bool drawn = xr_ok(inst->WaitSwapchainImage(sd->swapchain, &wait_info), "xrWaitSwapchainImage")
             && xr_vk_target_draw(sd->target, image_index);

   /* Release balances the acquire even on wait failure, or the image stays
    * acquired forever. */
   XrSwapchainImageReleaseInfo release_info { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
   xr_ok(inst->ReleaseSwapchainImage(sd->swapchain, &release_info), "xrReleaseSwapchainImage");

   /* Without a draw the image holds no fresh HUD, so submit the runtime's own
    * layers untouched rather than a quad over stale or uncleared contents. */
   if (!drawn)
      return end_frame_limited(inst, session, frameEndInfo);

   auto params = get_params_nonblocking();
   if (!params)
      return end_frame_limited(inst, session, frameEndInfo);
   XrCompositionLayerQuad quad { XR_TYPE_COMPOSITION_LAYER_QUAD };
   quad.layerFlags = XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT;
   quad.space = sd->space;
   quad.eyeVisibility = XR_EYE_VISIBILITY_BOTH;
   quad.subImage.swapchain = sd->swapchain;
   /* Show only the HUD window itself, wherever `position` put it on the
    * canvas, sized so its aspect is preserved. */
   int32_t content_x = 0, content_y = 0;
   uint32_t content_w = sd->width, content_h = sd->height;
   xr_vk_target_content_rect(sd->target, content_x, content_y, content_w, content_h);
   quad.subImage.imageRect.offset = { content_x, content_y };
   quad.subImage.imageRect.extent = { (int32_t)content_w, (int32_t)content_h };
   quad.subImage.imageArrayIndex = 0;
   quad.pose.orientation = { 0.f, 0.f, 0.f, 1.f };
   quad.pose.position = { params->vr_offset_x, params->vr_offset_y, -params->vr_distance };
   quad.size = { params->vr_size, params->vr_size * (float)content_h / (float)content_w };

   std::vector<const XrCompositionLayerBaseHeader *> layers(frameEndInfo->layers,
                                                            frameEndInfo->layers + frameEndInfo->layerCount);
   layers.push_back(reinterpret_cast<const XrCompositionLayerBaseHeader *>(&quad));
   if (!sd->announced) {
      SPDLOG_DEBUG("OpenXR HUD quad submitted: {:.2f}x{:.2f} m at {:.2f} m, anchor {}",
                   quad.size.width, quad.size.height, params->vr_distance, params->vr_anchor);
      sd->announced = true;
   }

   XrFrameEndInfo end_info = *frameEndInfo;
   end_info.layerCount = (uint32_t)layers.size();
   end_info.layers = layers.data();
   return end_frame_limited(inst, session, &end_info);
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
   if (!sd->vulkan) {
      SPDLOG_INFO("OpenXR session is not Vulkan, the HUD will not be drawn in the headset");
   } else if (!xr_vk_device_known(sd->binding.device)) {
      /* Rendering reuses the Vulkan layer's device tables, which only exist in
       * the copy of libMangoHud.so that intercepted vkCreateDevice. Decide it
       * here so no swapchain, space or stats are set up for nothing. */
      SPDLOG_WARN("OpenXR session's VkDevice was not seen by the Vulkan layer in this library "
                  "(MangoHud blacklisted for this app, or the Vulkan and OpenXR manifests point at "
                  "different libMangoHud.so files); not drawing the HUD in the headset");
      sd->vulkan = false;
   } else {
      SPDLOG_INFO("OpenXR Vulkan session created: VkDevice {} queue family {} index {}",
                   (void *)sd->binding.device, sd->binding.queueFamilyIndex, sd->binding.queueIndex);
   }

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
      if ((hook.always || (mangohud_enabled() && !is_blacklisted())) && strcmp(name, hook.name) == 0) {
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
   SPDLOG_INFO("MangoHud OpenXR layer loaded, negotiation succeeded");
   return XR_SUCCESS;
}
