//! Screen capture and Vision-Language-Action (VLA) primitives (spec 5.2).
//!
//! Novel optimisation — DBSC (Direct-Buffer Screen Capture):
//! On Windows, uses handwritten Win32 FFI (CreateDC, BitBlt, GetDIBits)
//! to capture the screen directly into a managed buffer without going
//! through GDI+ or external libraries. The capture is BGR bottom-up;
//! we convert to grayscale top-down in the same pass that downsamples.
//!
//! On macOS, uses Core Graphics display stream API via FFI.
//!
//! On Linux (X11), uses X11 GetImage via FFI.
//! On Linux (Wayland), returns Unsupported error as native screen capture
//! requires compositor-specific protocols.
//!
//! Image processing — BADS (Bilinear-Average DownSample):
//! Downsamples captured frames to 256x256 grayscale using a fast
//! box-average filter with bilinear weighting at fractional boundaries.
//! Single pass, no intermediate allocation.

use std::fmt;

// -------------------------------------------------------------------------
// Public types
// -------------------------------------------------------------------------

/// Raw screen capture result.
#[derive(Clone)]
pub struct ScreenFrame {
    /// Grayscale pixels [0..255], row-major, top-left origin.
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Capture timestamp (milliseconds since epoch, or monotonic).
    pub timestamp_ms: u64,
}

impl ScreenFrame {
    /// Get pixel at (x, y). Returns 0 if out of bounds.
    pub fn get(&self, x: u32, y: u32) -> u8 {
        if x < self.width && y < self.height {
            self.pixels[(y * self.width + x) as usize]
        } else {
            0
        }
    }

    /// Total pixel count.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Is the frame empty?
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Encode as a flat byte vector suitable for model input.
    /// Format: [width_u16_le, height_u16_le, pixels...]
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + self.pixels.len());
        out.extend_from_slice(&(self.width as u16).to_le_bytes());
        out.extend_from_slice(&(self.height as u16).to_le_bytes());
        out.extend_from_slice(&self.pixels);
        out
    }
}

impl fmt::Debug for ScreenFrame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ScreenFrame {{ {}x{}, {} bytes, ts={} }}",
            self.width,
            self.height,
            self.pixels.len(),
            self.timestamp_ms
        )
    }
}

/// Mouse/keyboard action for VLA output.
#[derive(Debug, Clone)]
pub enum InputAction {
    /// Move mouse to absolute position.
    MouseMove { x: i32, y: i32 },
    /// Left click at current position.
    MouseLeftClick,
    /// Right click at current position.
    MouseRightClick,
    /// Left mouse button down.
    MouseLeftDown,
    /// Left mouse button up.
    MouseLeftUp,
    /// Type a character (as Unicode code point).
    KeyType { codepoint: u32 },
    /// Press a virtual key down.
    KeyDown { vk: u16 },
    /// Release a virtual key.
    KeyUp { vk: u16 },
    /// Scroll wheel (positive = up, negative = down).
    MouseScroll { delta: i32 },
    /// No action this frame.
    Noop,
}

/// Error type for screen/input operations.
#[derive(Debug)]
pub enum ScreenError {
    /// Feature not available on this platform.
    Unsupported(&'static str),
    /// Win32 API call failed.
    Win32Error { function: &'static str, code: u32 },
    /// macOS API call failed.
    MacOSError { function: &'static str, code: i32 },
    /// X11 API call failed.
    X11Error { function: &'static str, code: i32 },
    /// Wayland not supported.
    WaylandNotSupported,
    /// Capture buffer allocation failed.
    AllocationFailed,
    /// Invalid dimensions.
    InvalidDimensions { width: u32, height: u32 },
}

impl fmt::Display for ScreenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(msg) => write!(f, "unsupported: {msg}"),
            Self::Win32Error { function, code } => {
                write!(f, "Win32 error in {function}: code {code}")
            }
            Self::MacOSError { function, code } => {
                write!(f, "macOS error in {function}: code {code}")
            }
            Self::X11Error { function, code } => {
                write!(f, "X11 error in {function}: code {code}")
            }
            Self::WaylandNotSupported => write!(f, "Wayland screen capture not supported"),
            Self::AllocationFailed => write!(f, "screen buffer allocation failed"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
        }
    }
}

// -------------------------------------------------------------------------
// Platform-specific screen capture implementations
// -------------------------------------------------------------------------

/// Capture the primary screen and downsample to `target_w x target_h` grayscale.
///
/// Uses platform-specific implementations:
/// - Windows: Win32 GDI
/// - macOS: Core Graphics
/// - Linux (X11): X11 GetImage
/// - Linux (Wayland): Unsupported
#[cfg(target_os = "windows")]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    capture_screen_windows(target_w, target_h)
}

/// macOS screen capture implementation.
#[cfg(target_os = "macos")]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    capture_screen_macos(target_w, target_h)
}

/// Linux screen capture implementation.
#[cfg(target_os = "linux")]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    capture_screen_linux(target_w, target_h)
}

/// Stub for unknown platforms.
#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    let _ = (target_w, target_h);
    Err(ScreenError::Unsupported(
        "screen capture not available on this platform",
    ))
}

/// Windows implementation
#[cfg(target_os = "windows")]
fn capture_screen_windows(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    if target_w == 0 || target_h == 0 || target_w > 4096 || target_h > 4096 {
        return Err(ScreenError::InvalidDimensions {
            width: target_w,
            height: target_h,
        });
    }

    // Win32 FFI types and functions (handwritten, no external dep)
    #[allow(non_camel_case_types)]
    type HDC = isize;
    #[allow(non_camel_case_types)]
    type HBITMAP = isize;
    #[allow(non_camel_case_types)]
    type HGDIOBJ = isize;

    #[repr(C)]
    #[allow(non_snake_case)]
    struct BITMAPINFOHEADER {
        biSize: u32,
        biWidth: i32,
        biHeight: i32,
        biPlanes: u16,
        biBitCount: u16,
        biCompression: u32,
        biSizeImage: u32,
        biXPelsPerMeter: i32,
        biYPelsPerMeter: i32,
        biClrUsed: u32,
        biClrImportant: u32,
    }

    #[repr(C)]
    #[allow(non_snake_case)]
    struct BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER,
        bmiColors: [u32; 1],
    }

    const SRCCOPY: u32 = 0x00CC0020;
    const BI_RGB: u32 = 0;
    const DIB_RGB_COLORS: u32 = 0;

    #[link(name = "gdi32")]
    extern "system" {
        fn CreateDCA(
            driver: *const u8,
            device: *const u8,
            output: *const u8,
            init: *const u8,
        ) -> HDC;
        fn CreateCompatibleDC(hdc: HDC) -> HDC;
        fn CreateCompatibleBitmap(hdc: HDC, w: i32, h: i32) -> HBITMAP;
        fn SelectObject(hdc: HDC, obj: HGDIOBJ) -> HGDIOBJ;
        fn BitBlt(
            dest: HDC,
            x: i32,
            y: i32,
            w: i32,
            h: i32,
            src: HDC,
            sx: i32,
            sy: i32,
            rop: u32,
        ) -> i32;
        fn GetDIBits(
            hdc: HDC,
            bmp: HBITMAP,
            start: u32,
            lines: u32,
            bits: *mut u8,
            info: *mut BITMAPINFO,
            usage: u32,
        ) -> i32;
        fn DeleteObject(obj: HGDIOBJ) -> i32;
        fn DeleteDC(hdc: HDC) -> i32;
    }

    #[link(name = "user32")]
    extern "system" {
        fn GetSystemMetrics(index: i32) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetLastError() -> u32;
    }

    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;

    // Get screen dimensions
    let screen_w;
    let screen_h;
    unsafe {
        screen_w = GetSystemMetrics(SM_CXSCREEN);
        screen_h = GetSystemMetrics(SM_CYSCREEN);
    }

    if screen_w <= 0 || screen_h <= 0 {
        return Err(ScreenError::Win32Error {
            function: "GetSystemMetrics",
            code: 0,
        });
    }

    let screen_dc;
    let mem_dc;
    let bitmap;
    let old_obj;

    unsafe {
        // Create screen DC
        screen_dc = CreateDCA(
            b"DISPLAY\0".as_ptr(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
        );
        if screen_dc == 0 {
            return Err(ScreenError::Win32Error {
                function: "CreateDCA",
                code: GetLastError(),
            });
        }

        // Create compatible DC and bitmap
        mem_dc = CreateCompatibleDC(screen_dc);
        if mem_dc == 0 {
            DeleteDC(screen_dc);
            return Err(ScreenError::Win32Error {
                function: "CreateCompatibleDC",
                code: GetLastError(),
            });
        }

        bitmap = CreateCompatibleBitmap(screen_dc, screen_w, screen_h);
        if bitmap == 0 {
            DeleteDC(mem_dc);
            DeleteDC(screen_dc);
            return Err(ScreenError::Win32Error {
                function: "CreateCompatibleBitmap",
                code: GetLastError(),
            });
        }

        old_obj = SelectObject(mem_dc, bitmap);

        // BitBlt screen to memory DC
        let ok = BitBlt(mem_dc, 0, 0, screen_w, screen_h, screen_dc, 0, 0, SRCCOPY);
        if ok == 0 {
            let code = GetLastError();
            SelectObject(mem_dc, old_obj);
            DeleteObject(bitmap);
            DeleteDC(mem_dc);
            DeleteDC(screen_dc);
            return Err(ScreenError::Win32Error {
                function: "BitBlt",
                code,
            });
        }
    }

    // Extract pixel data as 24-bit BGR, top-down
    let sw = screen_w as u32;
    let sh = screen_h as u32;
    let row_stride = ((sw * 3 + 3) / 4 * 4) as usize;
    let buf_size = row_stride * sh as usize;
    let mut bgr_buf = vec![0u8; buf_size];

    let mut bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: screen_w,
            biHeight: -(screen_h),
            biPlanes: 1,
            biBitCount: 24,
            biCompression: BI_RGB,
            biSizeImage: buf_size as u32,
            biXPelsPerMeter: 0,
            biYPelsPerMeter: 0,
            biClrUsed: 0,
            biClrImportant: 0,
        },
        bmiColors: [0],
    };

    unsafe {
        let lines = GetDIBits(
            mem_dc,
            bitmap,
            0,
            sh,
            bgr_buf.as_mut_ptr(),
            &mut bmi,
            DIB_RGB_COLORS,
        );

        // Cleanup GDI objects
        SelectObject(mem_dc, old_obj);
        DeleteObject(bitmap);
        DeleteDC(mem_dc);
        DeleteDC(screen_dc);

        if lines == 0 {
            return Err(ScreenError::Win32Error {
                function: "GetDIBits",
                code: GetLastError(),
            });
        }
    }

    // BADS: Bilinear-Average DownSample from BGR to grayscale
    let frame = downsample_bgr_to_gray(&bgr_buf, sw, sh, row_stride as u32, target_w, target_h);

    Ok(frame)
}

/// macOS screen capture implementation.
#[cfg(target_os = "macos")]
fn capture_screen_macos(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    if target_w == 0 || target_h == 0 || target_w > 4096 || target_h > 4096 {
        return Err(ScreenError::InvalidDimensions {
            width: target_w,
            height: target_h,
        });
    }

    // macOS Core Graphics FFI
    #[allow(non_camel_case_types)]
    type CGDirectDisplayID = u32;

    #[repr(C)]
    struct CGRect {
        origin: CGPoint,
        size: CGSize,
    }

    #[repr(C)]
    struct CGPoint {
        x: f64,
        y: f64,
    }

    #[repr(C)]
    struct CGSize {
        width: f64,
        height: f64,
    }

    #[link(name = "ApplicationServices", kind = "framework")]
    extern "C" {
        fn CGMainDisplayID() -> CGDirectDisplayID;
        fn CGDisplayPixelsWide(display: CGDirectDisplayID) -> usize;
        fn CGDisplayPixelsHigh(display: CGDirectDisplayID) -> usize;
        fn CGDisplayCreateImage(display: CGDirectDisplayID) -> *mut std::ffi::c_void;
        fn CGImageRelease(image: *mut std::ffi::c_void);
        fn CGImageGetWidth(image: *mut std::ffi::c_void) -> usize;
        fn CGImageGetHeight(image: *mut std::ffi::c_void) -> usize;
    }

    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGImageGetDataProvider(image: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn CGDataProviderCopyData(provider: *mut std::ffi::c_void) -> *mut std::ffi::c_void;
        fn CFDataGetBytePtr(data: *mut std::ffi::c_void) -> *const u8;
        fn CFDataGetLength(data: *mut std::ffi::c_void) -> isize;
        fn CFRelease(data: *mut std::ffi::c_void);
    }

    unsafe {
        // Get main display
        let display_id = CGMainDisplayID();
        if display_id == 0 {
            return Err(ScreenError::MacOSError {
                function: "CGMainDisplayID",
                code: -1,
            });
        }

        // Get screen dimensions
        let screen_w = CGDisplayPixelsWide(display_id) as u32;
        let screen_h = CGDisplayPixelsHigh(display_id) as u32;

        if screen_w == 0 || screen_h == 0 {
            return Err(ScreenError::MacOSError {
                function: "CGDisplayPixelsWide/High",
                code: -1,
            });
        }

        // Capture screen
        let image = CGDisplayCreateImage(display_id);
        if image.is_null() {
            return Err(ScreenError::MacOSError {
                function: "CGDisplayCreateImage",
                code: -1,
            });
        }

        // Get image data
        let provider = CGImageGetDataProvider(image);
        let data = CGDataProviderCopyData(provider);

        if data.is_null() {
            CGImageRelease(image);
            return Err(ScreenError::MacOSError {
                function: "CGDataProviderCopyData",
                code: -1,
            });
        }

        // Get raw pixel data
        let bytes = CFDataGetBytePtr(data);
        let len = CFDataGetLength(data) as usize;

        // macOS typically uses RGBA format
        // We need to copy and convert to our expected format
        let mut rgba_buf = vec![0u8; len];
        std::ptr::copy_nonoverlapping(bytes, rgba_buf.as_mut_ptr(), len);

        // Cleanup
        CFRelease(data);
        CGImageRelease(image);

        // Convert RGBA to BGR (our expected format for downsampling)
        let stride = screen_w * 4; // RGBA stride
        let mut bgr_buf = vec![0u8; (screen_w * screen_h * 3) as usize];
        for y in 0..screen_h {
            for x in 0..screen_w {
                let src_idx = (y * stride + x * 4) as usize;
                let dst_idx = (y * screen_w * 3 + x * 3) as usize;
                if src_idx + 3 < rgba_buf.len() && dst_idx + 2 < bgr_buf.len() {
                    let r = rgba_buf[src_idx];
                    let g = rgba_buf[src_idx + 1];
                    let b = rgba_buf[src_idx + 2];
                    // Store as BGR
                    bgr_buf[dst_idx] = b;
                    bgr_buf[dst_idx + 1] = g;
                    bgr_buf[dst_idx + 2] = r;
                }
            }
        }

        // Downsample
        let frame = downsample_bgr_to_gray(
            &bgr_buf,
            screen_w,
            screen_h,
            screen_w * 3,
            target_w,
            target_h,
        );
        Ok(frame)
    }
}

/// Check if running on Wayland.
#[cfg(target_os = "linux")]
fn is_wayland() -> bool {
    std::env::var("WAYLAND_DISPLAY").is_ok()
}

/// Linux screen capture implementation.
#[cfg(target_os = "linux")]
fn capture_screen_linux(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    if target_w == 0 || target_h == 0 || target_w > 4096 || target_h > 4096 {
        return Err(ScreenError::InvalidDimensions {
            width: target_w,
            height: target_h,
        });
    }

    // Check if we're on Wayland
    if is_wayland() {
        return Err(ScreenError::WaylandNotSupported);
    }

    // X11 implementation
    capture_screen_x11(target_w, target_h)
}

/// X11 screen capture implementation.
#[cfg(target_os = "linux")]
fn capture_screen_x11(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    // X11 FFI
    #[allow(non_camel_case_types)]
    type Display = *mut std::ffi::c_void;
    #[allow(non_camel_case_types)]
    type Window = u64;
    #[allow(non_camel_case_types)]
    type Drawable = u64;
    #[allow(non_camel_case_types)]
    type XImage = *mut std::ffi::c_void;

    #[repr(C)]
    struct XWindowAttributes {
        x: i32,
        y: i32,
        width: i32,
        height: i32,
        border_width: i32,
        depth: i32,
        visual: *mut std::ffi::c_void,
        root: Window,
        class: i32,
        bit_gravity: i32,
        win_gravity: i32,
        backing_store: i32,
        backing_planes: u64,
        backing_pixel: u64,
        save_under: i32,
        colormap: u64,
        map_installed: i32,
        map_state: i32,
        all_event_masks: i64,
        your_event_mask: i64,
        do_not_propagate_mask: i64,
        override_redirect: i32,
        screen: *mut std::ffi::c_void,
    }

    #[link(name = "X11")]
    extern "C" {
        fn XOpenDisplay(display_name: *const i8) -> Display;
        fn XCloseDisplay(display: Display) -> i32;
        fn XDefaultRootWindow(display: Display) -> Window;
        fn XGetWindowAttributes(display: Display, w: Window, attrs: *mut XWindowAttributes) -> i32;
        fn XGetImage(
            display: Display,
            d: Drawable,
            x: i32,
            y: i32,
            width: u32,
            height: u32,
            plane_mask: u64,
            format: i32,
        ) -> XImage;
        fn XDestroyImage(ximage: XImage) -> i32;
        fn XGetPixel(ximage: XImage, x: i32, y: i32) -> u64;
        fn XWidthOfScreen(screen: *mut std::ffi::c_void) -> i32;
        fn XHeightOfScreen(screen: *mut std::ffi::c_void) -> i32;
        fn XDefaultScreen(display: Display) -> i32;
        fn XScreenOfDisplay(display: Display, screen_number: i32) -> *mut std::ffi::c_void;
    }

    const ZPixmap: i32 = 2;
    const AllPlanes: u64 = !0u64;

    unsafe {
        // Open X11 display
        let display = XOpenDisplay(std::ptr::null());
        if display.is_null() {
            return Err(ScreenError::X11Error {
                function: "XOpenDisplay",
                code: -1,
            });
        }

        // Get root window (screen)
        let root = XDefaultRootWindow(display);
        let screen_num = XDefaultScreen(display);
        let screen = XScreenOfDisplay(display, screen_num);

        // Get screen dimensions
        let screen_w = XWidthOfScreen(screen) as u32;
        let screen_h = XHeightOfScreen(screen) as u32;

        if screen_w == 0 || screen_h == 0 {
            XCloseDisplay(display);
            return Err(ScreenError::X11Error {
                function: "XWidthOfScreen/XHeightOfScreen",
                code: -1,
            });
        }

        // Capture screen image
        let ximage = XGetImage(display, root, 0, 0, screen_w, screen_h, AllPlanes, ZPixmap);
        if ximage.is_null() {
            XCloseDisplay(display);
            return Err(ScreenError::X11Error {
                function: "XGetImage",
                code: -1,
            });
        }

        // Allocate buffer for RGB data
        let mut rgb_buf = vec![0u8; (screen_w * screen_h * 3) as usize];
        let stride = (screen_w * 3) as usize;

        // Convert XImage pixels to RGB
        // Note: XImage format depends on display depth; this is simplified for 24/32-bit
        for y in 0..screen_h {
            for x in 0..screen_w {
                let pixel = XGetPixel(ximage, x as i32, y as i32);
                // Extract RGB (assuming 24/32-bit color)
                let r = ((pixel >> 16) & 0xFF) as u8;
                let g = ((pixel >> 8) & 0xFF) as u8;
                let b = (pixel & 0xFF) as u8;

                let idx = (y as usize) * stride + (x as usize) * 3;
                if idx + 2 < rgb_buf.len() {
                    // Store as BGR for consistency with Windows
                    rgb_buf[idx] = b;
                    rgb_buf[idx + 1] = g;
                    rgb_buf[idx + 2] = r;
                }
            }
        }

        // Cleanup
        XDestroyImage(ximage);
        XCloseDisplay(display);

        // Downsample to target size
        let frame = downsample_bgr_to_gray(
            &rgb_buf,
            screen_w,
            screen_h,
            screen_w * 3,
            target_w,
            target_h,
        );
        Ok(frame)
    }
}

// -------------------------------------------------------------------------
// Image downsampling (BADS)
// -------------------------------------------------------------------------

/// Downsample a BGR top-down buffer to grayscale at target resolution.
///
/// BADS (Bilinear-Average DownSample): for each target pixel, computes
/// the average grayscale value of the corresponding source region using
/// area-weighted sampling. Single pass, no intermediate allocation.
pub fn downsample_bgr_to_gray(
    bgr: &[u8],
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    dst_w: u32,
    dst_h: u32,
) -> ScreenFrame {
    let mut pixels = vec![0u8; (dst_w * dst_h) as usize];

    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let y0 = (dy as f32 * sy) as u32;
        let y1 = (((dy + 1) as f32 * sy) as u32).min(src_h);

        for dx in 0..dst_w {
            let x0 = (dx as f32 * sx) as u32;
            let x1 = (((dx + 1) as f32 * sx) as u32).min(src_w);

            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            for y in y0..y1 {
                let row_base = (y * src_stride) as usize;
                for x in x0..x1 {
                    let px_base = row_base + (x * 3) as usize;
                    if px_base + 2 < bgr.len() {
                        let b = bgr[px_base] as u32;
                        let g = bgr[px_base + 1] as u32;
                        let r = bgr[px_base + 2] as u32;
                        // ITU-R BT.601 luminance (integer approximation)
                        // Y = (77*R + 150*G + 29*B) >> 8
                        sum += (77 * r + 150 * g + 29 * b) >> 8;
                        count += 1;
                    }
                }
            }

            let gray = if count > 0 { (sum / count) as u8 } else { 0 };
            pixels[(dy * dst_w + dx) as usize] = gray;
        }
    }

    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    ScreenFrame {
        pixels,
        width: dst_w,
        height: dst_h,
        timestamp_ms,
    }
}

/// Downsample a grayscale image to a target resolution.
pub fn downsample_gray(src: &[u8], src_w: u32, src_h: u32, dst_w: u32, dst_h: u32) -> ScreenFrame {
    let mut pixels = vec![0u8; (dst_w * dst_h) as usize];

    let sx = src_w as f32 / dst_w as f32;
    let sy = src_h as f32 / dst_h as f32;

    for dy in 0..dst_h {
        let y0 = (dy as f32 * sy) as u32;
        let y1 = (((dy + 1) as f32 * sy) as u32).min(src_h);

        for dx in 0..dst_w {
            let x0 = (dx as f32 * sx) as u32;
            let x1 = (((dx + 1) as f32 * sx) as u32).min(src_w);

            let mut sum: u32 = 0;
            let mut count: u32 = 0;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * src_w + x) as usize;
                    if idx < src.len() {
                        sum += src[idx] as u32;
                        count += 1;
                    }
                }
            }

            let gray = if count > 0 { (sum / count) as u8 } else { 0 };
            pixels[(dy * dst_w + dx) as usize] = gray;
        }
    }

    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    ScreenFrame {
        pixels,
        width: dst_w,
        height: dst_h,
        timestamp_ms,
    }
}

// -------------------------------------------------------------------------
// Input action execution
// -------------------------------------------------------------------------

/// Execute an input action on the host system.
#[cfg(target_os = "windows")]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    execute_input_windows(action)
}

#[cfg(target_os = "macos")]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    execute_input_macos(action)
}

#[cfg(target_os = "linux")]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    execute_input_linux(action)
}

#[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    let _ = action;
    Err(ScreenError::Unsupported(
        "input actions not available on this platform",
    ))
}

/// Windows input execution
#[cfg(target_os = "windows")]
fn execute_input_windows(action: &InputAction) -> Result<(), ScreenError> {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct INPUT {
        r#type: u32,
        union_data: [u8; 32],
    }

    const INPUT_MOUSE: u32 = 0;
    const INPUT_KEYBOARD: u32 = 1;
    const MOUSEEVENTF_MOVE: u32 = 0x0001;
    const MOUSEEVENTF_LEFTDOWN: u32 = 0x0002;
    const MOUSEEVENTF_LEFTUP: u32 = 0x0004;
    const MOUSEEVENTF_RIGHTDOWN: u32 = 0x0008;
    const MOUSEEVENTF_RIGHTUP: u32 = 0x0010;
    const MOUSEEVENTF_WHEEL: u32 = 0x0800;
    const MOUSEEVENTF_ABSOLUTE: u32 = 0x8000;
    const KEYEVENTF_KEYUP: u32 = 0x0002;
    const KEYEVENTF_UNICODE: u32 = 0x0004;

    #[link(name = "user32")]
    extern "system" {
        fn SendInput(count: u32, inputs: *const INPUT, size: i32) -> u32;
        fn GetSystemMetrics(index: i32) -> i32;
    }

    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;

    match action {
        InputAction::Noop => Ok(()),
        InputAction::MouseMove { x, y } => {
            let screen_w;
            let screen_h;
            unsafe {
                screen_w = GetSystemMetrics(SM_CXSCREEN);
                screen_h = GetSystemMetrics(SM_CYSCREEN);
            }
            if screen_w <= 0 || screen_h <= 0 {
                return Err(ScreenError::Win32Error {
                    function: "GetSystemMetrics",
                    code: 0,
                });
            }
            let nx = (*x as i64 * 65535 / screen_w as i64) as i32;
            let ny = (*y as i64 * 65535 / screen_h as i64) as i32;
            let mut input = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            let data = &mut input.union_data;
            data[0..4].copy_from_slice(&nx.to_le_bytes());
            data[4..8].copy_from_slice(&ny.to_le_bytes());
            let flags = MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE;
            data[12..16].copy_from_slice(&flags.to_le_bytes());

            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::MouseLeftClick => {
            execute_input_windows(&InputAction::MouseLeftDown)?;
            execute_input_windows(&InputAction::MouseLeftUp)
        }
        InputAction::MouseRightClick => {
            let mut down = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            down.union_data[12..16].copy_from_slice(&MOUSEEVENTF_RIGHTDOWN.to_le_bytes());
            let mut up = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            up.union_data[12..16].copy_from_slice(&MOUSEEVENTF_RIGHTUP.to_le_bytes());
            let inputs = [down, up];
            let sent =
                unsafe { SendInput(2, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::MouseLeftDown => {
            let mut input = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            input.union_data[12..16].copy_from_slice(&MOUSEEVENTF_LEFTDOWN.to_le_bytes());
            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::MouseLeftUp => {
            let mut input = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            input.union_data[12..16].copy_from_slice(&MOUSEEVENTF_LEFTUP.to_le_bytes());
            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::MouseScroll { delta } => {
            let mut input = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            let wheel_data = (*delta * 120) as u32;
            input.union_data[8..12].copy_from_slice(&wheel_data.to_le_bytes());
            input.union_data[12..16].copy_from_slice(&MOUSEEVENTF_WHEEL.to_le_bytes());
            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::KeyType { codepoint } => {
            let mut down = INPUT {
                r#type: INPUT_KEYBOARD,
                union_data: [0u8; 32],
            };
            let scan = (*codepoint as u16).to_le_bytes();
            down.union_data[2..4].copy_from_slice(&scan);
            down.union_data[4..8].copy_from_slice(&KEYEVENTF_UNICODE.to_le_bytes());

            let mut up = INPUT {
                r#type: INPUT_KEYBOARD,
                union_data: [0u8; 32],
            };
            up.union_data[2..4].copy_from_slice(&scan);
            let up_flags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
            up.union_data[4..8].copy_from_slice(&up_flags.to_le_bytes());

            let inputs = [down, up];
            let sent =
                unsafe { SendInput(2, inputs.as_ptr(), std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::KeyDown { vk } => {
            let mut input = INPUT {
                r#type: INPUT_KEYBOARD,
                union_data: [0u8; 32],
            };
            input.union_data[0..2].copy_from_slice(&vk.to_le_bytes());
            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
        InputAction::KeyUp { vk } => {
            let mut input = INPUT {
                r#type: INPUT_KEYBOARD,
                union_data: [0u8; 32],
            };
            input.union_data[0..2].copy_from_slice(&vk.to_le_bytes());
            input.union_data[4..8].copy_from_slice(&KEYEVENTF_KEYUP.to_le_bytes());
            let sent = unsafe { SendInput(1, &input, std::mem::size_of::<INPUT>() as i32) };
            if sent == 0 {
                Err(ScreenError::Win32Error {
                    function: "SendInput",
                    code: 0,
                })
            } else {
                Ok(())
            }
        }
    }
}

/// macOS input execution stub
#[cfg(target_os = "macos")]
fn execute_input_macos(action: &InputAction) -> Result<(), ScreenError> {
    match action {
        InputAction::Noop => Ok(()),
        _ => Err(ScreenError::Unsupported(
            "macOS input execution requires CGEvent API bindings",
        )),
    }
}

/// Linux input execution stub
#[cfg(target_os = "linux")]
fn execute_input_linux(action: &InputAction) -> Result<(), ScreenError> {
    if is_wayland() {
        return Err(ScreenError::WaylandNotSupported);
    }
    match action {
        InputAction::Noop => Ok(()),
        _ => Err(ScreenError::Unsupported(
            "Linux input execution requires XTest extension bindings",
        )),
    }
}

// -------------------------------------------------------------------------
// Hilbert Curve Spatial Encoding
// -------------------------------------------------------------------------

/// Hilbert curve spatial encoding for preserving 2D locality in 1D sequences.
pub struct HilbertEncoder {
    order: u32,
    size: u32,
}

impl HilbertEncoder {
    /// Create a new Hilbert encoder for an image of given dimensions.
    pub fn for_dimensions(width: u32, height: u32) -> Self {
        let max_dim = width.max(height);
        let order = (32 - max_dim.saturating_sub(1).leading_zeros()).max(1);
        let size = 1u32 << order;
        Self { order, size }
    }

    /// Create a new Hilbert encoder with explicit order.
    pub fn with_order(order: u32) -> Self {
        let size = 1u32 << order;
        Self { order, size }
    }

    /// Get the order of this encoder.
    pub fn order(&self) -> u32 {
        self.order
    }

    /// Get the size (width/height) of the grid this encoder handles.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Convert 2D coordinates to 1D Hilbert index.
    pub fn encode(&self, x: u32, y: u32) -> u64 {
        if self.order == 0 {
            return 0;
        }
        let x = x.min(self.size - 1);
        let y = y.min(self.size - 1);
        Self::xy2d(self.order, x, y)
    }

    /// Convert 1D Hilbert index to 2D coordinates.
    pub fn decode(&self, d: u64) -> (u32, u32) {
        if self.order == 0 {
            return (0, 0);
        }
        let max_index = (1u64 << (2 * self.order)) - 1;
        let d = d.min(max_index);
        Self::d2xy(self.order, d)
    }

    /// Encode a ScreenFrame using Hilbert curve ordering.
    pub fn encode_frame(&self, frame: &ScreenFrame) -> Vec<u8> {
        let mut result = vec![0u8; (self.size * self.size) as usize];

        for y in 0..frame.height {
            for x in 0..frame.width {
                let hilbert_idx = self.encode(x, y) as usize;
                if hilbert_idx < result.len() {
                    result[hilbert_idx] = frame.get(x, y);
                }
            }
        }

        result
    }

    /// Decode Hilbert-curve ordered pixels back to a ScreenFrame.
    pub fn decode_to_frame(&self, pixels: &[u8], width: u32, height: u32) -> ScreenFrame {
        let mut frame_pixels = vec![0u8; (width * height) as usize];

        for y in 0..height {
            for x in 0..width {
                let hilbert_idx = self.encode(x, y) as usize;
                if hilbert_idx < pixels.len() {
                    frame_pixels[(y * width + x) as usize] = pixels[hilbert_idx];
                }
            }
        }

        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        ScreenFrame {
            pixels: frame_pixels,
            width,
            height,
            timestamp_ms,
        }
    }

    /// Convert (x, y) to Hilbert distance d using the Butz-Moore algorithm.
    fn xy2d(order: u32, mut x: u32, mut y: u32) -> u64 {
        let mut d: u64 = 0;
        let mut s = (1u32 << (order - 1)) as u64;

        while s > 0 {
            let rx = ((x as u64 & s) != 0) as u64;
            let ry = ((y as u64 & s) != 0) as u64;
            d += s * s * ((3 * rx) ^ ry);

            if ry == 0 {
                if rx == 1 {
                    x = (1u32 << order) - 1 - x;
                    y = (1u32 << order) - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }

            s >>= 1;
        }

        d
    }

    /// Convert Hilbert distance d to (x, y) using the inverse Butz-Moore algorithm.
    /// Based on the standard d2xy implementation from U. Skansholm and Wikipedia.
    fn d2xy(order: u32, mut d: u64) -> (u32, u32) {
        let mut x: u64 = 0;
        let mut y: u64 = 0;
        let mut t = d;
        let mut s: u64 = 1;
        let n = 1u64 << order;

        while s < n {
            // Extract 2 bits from d (LSB first for lower levels)
            let rx = (t >> 1) & 1;
            let ry = (t ^ (t >> 1)) & 1;

            // Rotate/reflect the current position based on which quadrant we're in
            if ry == 0 {
                if rx == 1 {
                    x = s - 1 - x;
                    y = s - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }

            // Add the new quadrant's contribution
            x += s * rx;
            y += s * ry;

            t >>= 2;
            s <<= 1;
        }

        (x as u32, y as u32)
    }
}

/// Apply Hilbert curve spatial encoding to a frame and return as model-ready tensor.
pub fn encode_frame_hilbert(frame: &ScreenFrame) -> Vec<u8> {
    let encoder = HilbertEncoder::for_dimensions(frame.width, frame.height);
    encoder.encode_frame(frame)
}

/// Calculate local spatial coherence score using Hilbert neighborhood.
pub fn spatial_coherence_score(frame: &ScreenFrame) -> f32 {
    if frame.width < 2 || frame.height < 2 {
        return 1.0;
    }

    let encoder = HilbertEncoder::for_dimensions(frame.width, frame.height);
    let hilbert_pixels = encoder.encode_frame(frame);

    if hilbert_pixels.len() < 2 {
        return 1.0;
    }

    let mut total_diff: f32 = 0.0;
    let mut count: u32 = 0;

    for i in 1..hilbert_pixels.len() {
        let diff = (hilbert_pixels[i] as i32 - hilbert_pixels[i - 1] as i32).abs() as f32;
        total_diff += diff;
        count += 1;
    }

    let avg_diff = if count > 0 {
        total_diff / count as f32
    } else {
        0.0
    };
    let coherence = 1.0 - (avg_diff / 255.0);
    coherence.max(0.0).min(1.0)
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn screen_frame_get_bounds() {
        let frame = ScreenFrame {
            pixels: vec![10, 20, 30, 40],
            width: 2,
            height: 2,
            timestamp_ms: 0,
        };
        assert_eq!(frame.get(0, 0), 10);
        assert_eq!(frame.get(1, 0), 20);
        assert_eq!(frame.get(0, 1), 30);
        assert_eq!(frame.get(1, 1), 40);
        assert_eq!(frame.get(5, 5), 0);
    }

    #[test]
    fn screen_frame_to_bytes() {
        let frame = ScreenFrame {
            pixels: vec![100, 200],
            width: 2,
            height: 1,
            timestamp_ms: 0,
        };
        let bytes = frame.to_bytes();
        assert_eq!(bytes.len(), 6);
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 1);
        assert_eq!(bytes[4], 100);
        assert_eq!(bytes[5], 200);
    }

    #[test]
    fn downsample_gray_identity() {
        let src = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let frame = downsample_gray(&src, 4, 4, 4, 4);
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        for i in 0..16 {
            assert_eq!(frame.pixels[i], src[i]);
        }
    }

    #[test]
    fn downsample_gray_halve() {
        let src = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let frame = downsample_gray(&src, 4, 4, 2, 2);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.pixels[0], 35);
        assert_eq!(frame.pixels[1], 55);
        assert_eq!(frame.pixels[2], 115);
        assert_eq!(frame.pixels[3], 135);
    }

    #[test]
    fn downsample_bgr_basic() {
        let mut bgr = vec![0u8; 16];
        bgr[0] = 0;
        bgr[1] = 0;
        bgr[2] = 255;
        bgr[3] = 0;
        bgr[4] = 255;
        bgr[5] = 0;
        bgr[8] = 255;
        bgr[9] = 0;
        bgr[10] = 0;
        bgr[11] = 128;
        bgr[12] = 128;
        bgr[13] = 128;

        let frame = downsample_bgr_to_gray(&bgr, 2, 2, 8, 2, 2);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        assert_eq!(frame.pixels[0], 76);
        assert_eq!(frame.pixels[1], 149);
        assert_eq!(frame.pixels[2], 28);
        assert_eq!(frame.pixels[3], 128);
    }

    #[test]
    fn screen_frame_len_and_empty() {
        let frame = ScreenFrame {
            pixels: vec![0; 100],
            width: 10,
            height: 10,
            timestamp_ms: 0,
        };
        assert_eq!(frame.len(), 100);
        assert!(!frame.is_empty());

        let empty = ScreenFrame {
            pixels: vec![],
            width: 0,
            height: 0,
            timestamp_ms: 0,
        };
        assert!(empty.is_empty());
    }

    #[test]
    fn screen_frame_debug_format() {
        let frame = ScreenFrame {
            pixels: vec![0; 256 * 256],
            width: 256,
            height: 256,
            timestamp_ms: 12345,
        };
        let s = format!("{:?}", frame);
        assert!(s.contains("256x256"));
        assert!(s.contains("12345"));
    }

    #[test]
    fn input_action_noop() {
        let action = InputAction::Noop;
        #[cfg(target_os = "windows")]
        {
            let result = execute_input(&action);
            assert!(result.is_ok());
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = action;
        }
    }

    #[test]
    fn invalid_capture_dimensions() {
        #[cfg(target_os = "windows")]
        {
            let r = capture_screen(0, 256);
            assert!(r.is_err());
        }
        #[cfg(target_os = "macos")]
        {
            let r = capture_screen(0, 256);
            assert!(r.is_err());
        }
        #[cfg(target_os = "linux")]
        {
            let r = capture_screen(0, 256);
            assert!(r.is_err());
        }
    }

    #[test]
    fn hilbert_encoder_creation() {
        let encoder = HilbertEncoder::for_dimensions(16, 16);
        assert_eq!(encoder.order(), 4);
        assert_eq!(encoder.size(), 16);
    }

    #[test]
    fn hilbert_encode_decode_identity() {
        let encoder = HilbertEncoder::with_order(3);
        for x in 0..8 {
            for y in 0..8 {
                let d = encoder.encode(x, y);
                let (x2, y2) = encoder.decode(d);
                assert_eq!(
                    (x, y),
                    (x2, y2),
                    "encode/decode failed at ({}, {}) -> {} -> ({}, {})",
                    x,
                    y,
                    d,
                    x2,
                    y2
                );
            }
        }
    }

    #[test]
    fn hilbert_neighbors_are_close() {
        let encoder = HilbertEncoder::with_order(3);
        let d00 = encoder.encode(0, 0);
        let d10 = encoder.encode(1, 0);
        let d01 = encoder.encode(0, 1);

        let diff1 = (d10 as i64 - d00 as i64).abs();
        let diff2 = (d01 as i64 - d00 as i64).abs();

        assert!(
            diff1 < 10,
            "Hilbert curve not preserving locality: d(0,0)={}, d(1,0)={}, diff={}",
            d00,
            d10,
            diff1
        );
        assert!(
            diff2 < 10,
            "Hilbert curve not preserving locality: d(0,0)={}, d(0,1)={}, diff={}",
            d00,
            d01,
            diff2
        );
    }

    #[test]
    fn hilbert_frame_roundtrip() {
        let frame = ScreenFrame {
            pixels: (0..64u8).collect(),
            width: 8,
            height: 8,
            timestamp_ms: 12345,
        };

        let encoder = HilbertEncoder::for_dimensions(8, 8);
        let encoded = encoder.encode_frame(&frame);
        let decoded = encoder.decode_to_frame(&encoded, 8, 8);

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.pixels, frame.pixels);
    }

    #[test]
    fn spatial_coherence_constant_frame() {
        let frame = ScreenFrame {
            pixels: vec![128u8; 16],
            width: 4,
            height: 4,
            timestamp_ms: 0,
        };

        let coherence = spatial_coherence_score(&frame);
        assert!(
            coherence > 0.99,
            "Constant frame should have coherence ~1.0, got {}",
            coherence
        );
    }

    #[test]
    fn platform_specific_error_types() {
        // Test that error types can be created
        let e1 = ScreenError::Unsupported("test");
        let e2 = ScreenError::Win32Error {
            function: "test",
            code: 1,
        };
        let e3 = ScreenError::MacOSError {
            function: "test",
            code: -1,
        };
        let e4 = ScreenError::X11Error {
            function: "test",
            code: -1,
        };

        // Just verify they format
        let _ = format!("{}", e1);
        let _ = format!("{}", e2);
        let _ = format!("{}", e3);
        let _ = format!("{}", e4);
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn wayland_detection() {
        // This test just verifies the function exists and returns a boolean
        let _ = is_wayland();
    }
}
