use super::{Arc, ImageBuffer, ImageFrame, RenderImage, Rgba};
use crate::{pipeline::skeleton, types::Frame};

pub(super) fn frame_to_image(
    frame: &Frame,
    overlay: Option<&[(f32, f32)]>,
) -> Option<Arc<RenderImage>> {
    let mut rgba = frame.rgba.clone();
    if let Some(points) = overlay {
        skeleton::draw_skeleton(&mut rgba, frame.width, frame.height, points);
    }

    for px in rgba.chunks_exact_mut(4) {
        px.swap(0, 2);
    }

    let buffer = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(frame.width, frame.height, rgba)?;
    let frame = ImageFrame::new(buffer);

    Some(Arc::new(RenderImage::new(vec![frame])))
}
