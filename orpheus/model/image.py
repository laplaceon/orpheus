def resize_images(images,
                  size,
                  method=ResizeMethodV1.BILINEAR,
                  align_corners=False,
                  preserve_aspect_ratio=False,
                  name=None):
  """Resize `images` to `size` using the specified `method`.
  Resized images will be distorted if their original aspect ratio is not
  the same as `size`.  To avoid distortions see
  `tf.image.resize_with_pad` or `tf.image.resize_with_crop_or_pad`.
  The `method` can be one of:
  *   <b>`tf.image.ResizeMethod.BILINEAR`</b>: [Bilinear interpolation.](
    https://en.wikipedia.org/wiki/Bilinear_interpolation)
  *   <b>`tf.image.ResizeMethod.NEAREST_NEIGHBOR`</b>: [
    Nearest neighbor interpolation.](
    https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation)
  *   <b>`tf.image.ResizeMethod.BICUBIC`</b>: [Bicubic interpolation.](
    https://en.wikipedia.org/wiki/Bicubic_interpolation)
  *   <b>`tf.image.ResizeMethod.AREA`</b>: Area interpolation.
  The return value has the same type as `images` if `method` is
  `tf.image.ResizeMethod.NEAREST_NEIGHBOR`. It will also have the same type
  as `images` if the size of `images` can be statically determined to be the
  same as `size`, because `images` is returned in this case. Otherwise, the
  return value has type `float32`.
  Args:
    images: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    size: A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The new
      size for the images.
    method: ResizeMethod.  Defaults to `tf.image.ResizeMethod.BILINEAR`.
    align_corners: bool.  If True, the centers of the 4 corner pixels of the
      input and output tensors are aligned, preserving the values at the corner
      pixels. Defaults to `False`.
    preserve_aspect_ratio: Whether to preserve the aspect ratio. If this is set,
      then `images` will be resized to a size that fits in `size` while
      preserving the aspect ratio of the original image. Scales up the image if
      `size` is bigger than the current size of the `image`. Defaults to False.
    name: A name for this operation (optional).
  Raises:
    ValueError: if the shape of `images` is incompatible with the
      shape arguments to this function
    ValueError: if `size` has invalid shape or type.
    ValueError: if an unsupported resize method is specified.
  Returns:
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
  """

  def resize_fn(images_t, new_size):
    """Legacy resize core function, passed to _resize_images_common."""
    if method == ResizeMethodV1.BILINEAR or method == ResizeMethod.BILINEAR:
      return gen_image_ops.resize_bilinear(
          images_t, new_size, align_corners=align_corners)
    elif (method == ResizeMethodV1.NEAREST_NEIGHBOR or
          method == ResizeMethod.NEAREST_NEIGHBOR):
      return gen_image_ops.resize_nearest_neighbor(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.BICUBIC or method == ResizeMethod.BICUBIC:
      return gen_image_ops.resize_bicubic(
          images_t, new_size, align_corners=align_corners)
    elif method == ResizeMethodV1.AREA or method == ResizeMethod.AREA:
      return gen_image_ops.resize_area(
          images_t, new_size, align_corners=align_corners)
    else:
      raise ValueError('Resize method is not implemented: {}'.format(method))

  return _resize_images_common(
      images,
      resize_fn,
      size,
      preserve_aspect_ratio=preserve_aspect_ratio,
      name=name,
      skip_resize_if_same=True)
