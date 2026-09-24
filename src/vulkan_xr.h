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

xr_vk_target *xr_vk_target_create(const xr_vk_target_info& info);
/* Advances frame stats and lays out the HUD. Returns true when there is
 * something to draw this frame. */
bool xr_vk_target_update(xr_vk_target *target);
/* Records and submits the HUD draw into images[image_index]. The image must
 * be in VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL and is left in that layout. */
void xr_vk_target_draw(xr_vk_target *target, uint32_t image_index);
void xr_vk_target_destroy(xr_vk_target *target);

#endif //MANGOHUD_VULKAN_XR_H
