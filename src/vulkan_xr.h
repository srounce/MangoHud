#pragma once
#ifndef MANGOHUD_VULKAN_XR_H
#define MANGOHUD_VULKAN_XR_H

#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

/* Overlay rendering into images owned by an OpenXR swapchain. The Vulkan
 * layer keeps the dispatch tables and stats, so this only works for a
 * VkDevice that layer has already seen. */

struct xr_vk_target;

struct xr_vk_target_info {
   VkInstance instance;
   VkDevice device;
   uint32_t queue_family_index;
   uint32_t queue_index;
   VkFormat format;
   uint32_t width;
   uint32_t height;
   std::vector<VkImage> images;
};

/* Whether the Vulkan layer in this library intercepted the creation of
 * `device`. Rendering reuses its dispatch tables, so a session bound to an
 * unknown device cannot be drawn and should not be set up at all. */
bool xr_vk_device_known(VkDevice device);

xr_vk_target *xr_vk_target_create(const xr_vk_target_info& info);
/* Advances frame stats and lays out the HUD. Returns true when there is
 * something to draw this frame. */
bool xr_vk_target_update(xr_vk_target *target);
/* Records and submits the HUD draw into images[image_index]. The image must
 * be in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL and is left in that layout.
 * Returns false if nothing was recorded, in which case the image was not
 * cleared and must not be presented. */
bool xr_vk_target_draw(xr_vk_target *target, uint32_t image_index);
/* Pixel rectangle of the HUD window this frame (valid after update), so the
 * caller can crop the composition layer to the content instead of the whole
 * texture. Clamped to the texture. */
void xr_vk_target_content_rect(xr_vk_target *target, int32_t& x, int32_t& y, uint32_t& width, uint32_t& height);
void xr_vk_target_destroy(xr_vk_target *target);

#endif //MANGOHUD_VULKAN_XR_H
