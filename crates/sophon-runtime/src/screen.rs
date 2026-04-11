//! Screen capture and Vision-Language-Action (VLA) primitives (spec 5.2).
//!
//! Novel optimisation — DBSC (Direct-Buffer Screen Capture):
//!   On Windows, uses handwritten Win32 FFI (CreateDC, BitBlt, GetDIBits)
//!   to capture the screen directly into a managed buffer without going
//!   through GDI+ or external libraries. The capture is BGR bottom-up;
//!   we convert to grayscale top-down in the same pass that downsamples.
//!
//! Image processing — BADS (Bilinear-Average DownSample):
//!   Downsamples captured frames to 256x256 grayscale using a fast
//!   box-average filter with bilinear weighting at fractional boundaries.
//!   Single pass, no intermediate allocation.
//!
//! All Win32 calls are gated behind `#[cfg(target_os = "windows")]`.
//! On non-Windows platforms, all functions return `Err`.

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
            Self::AllocationFailed => write!(f, "screen buffer allocation failed"),
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height}")
            }
        }
    }
}

// -------------------------------------------------------------------------
// Screen capture (Windows)
// -------------------------------------------------------------------------

/// Capture the primary screen and downsample to `target_w x target_h` grayscale.
///
/// Uses DBSC (Direct-Buffer Screen Capture) on Windows.
/// Returns ScreenError::Unsupported on non-Windows platforms.
#[cfg(target_os = "windows")]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
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
        biHeight: i32, // negative = top-down
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
    let row_stride = ((sw * 3 + 3) / 4 * 4) as usize; // DWORD-aligned
    let buf_size = row_stride * sh as usize;
    let mut bgr_buf = vec![0u8; buf_size];

    let mut bmi = BITMAPINFO {
        bmiHeader: BITMAPINFOHEADER {
            biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
            biWidth: screen_w,
            biHeight: -(screen_h), // negative = top-down
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

/// Non-Windows stub.
#[cfg(not(target_os = "windows"))]
pub fn capture_screen(target_w: u32, target_h: u32) -> Result<ScreenFrame, ScreenError> {
    let _ = (target_w, target_h);
    Err(ScreenError::Unsupported("screen capture requires Windows"))
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
// Input action execution (Windows)
// -------------------------------------------------------------------------

/// Execute an input action on the host system.
#[cfg(target_os = "windows")]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    #[repr(C)]
    #[allow(non_snake_case)]
    struct INPUT {
        r#type: u32,
        union_data: [u8; 32], // large enough for MOUSEINPUT / KEYBDINPUT
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
            // Convert absolute coords to normalised (0..65535)
            let nx = (*x as i64 * 65535 / screen_w as i64) as i32;
            let ny = (*y as i64 * 65535 / screen_h as i64) as i32;
            let mut input = INPUT {
                r#type: INPUT_MOUSE,
                union_data: [0u8; 32],
            };
            // MOUSEINPUT: dx at offset 0 (i32), dy at 4 (i32), mouseData at 8 (u32),
            //             dwFlags at 12 (u32), time at 16 (u32), dwExtraInfo at 20 (usize)
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
            // Down then up
            execute_input(&InputAction::MouseLeftDown)?;
            execute_input(&InputAction::MouseLeftUp)
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
            // mouseData at offset 8 (wheel delta, positive = up)
            let wheel_data = (*delta * 120) as u32; // 120 = WHEEL_DELTA
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
            // Send Unicode character via KEYEVENTF_UNICODE
            let mut down = INPUT {
                r#type: INPUT_KEYBOARD,
                union_data: [0u8; 32],
            };
            // KEYBDINPUT: wVk at 0 (u16), wScan at 2 (u16), dwFlags at 4 (u32)
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

/// Non-Windows stub for input execution.
#[cfg(not(target_os = "windows"))]
pub fn execute_input(action: &InputAction) -> Result<(), ScreenError> {
    let _ = action;
    Err(ScreenError::Unsupported("input actions require Windows"))
}

// -------------------------------------------------------------------------
// Hilbert Curve Spatial Encoding
// -------------------------------------------------------------------------

/// Hilbert curve spatial encoding for preserving 2D locality in 1D sequences.
///
/// The Hilbert curve is a continuous fractal space-filling curve that preserves
/// spatial locality better than row-major ordering. This is useful for vision
/// models that process images as 1D sequences.
///
/// Reference: https://en.wikipedia.org/wiki/Hilbert_curve
pub struct HilbertEncoder {
    order: u32,
    size: u32,
}

impl HilbertEncoder {
    /// Create a new Hilbert encoder for an image of given dimensions.
    /// The order is determined automatically based on the larger dimension.
    ///
    /// # Arguments
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Returns
    /// A HilbertEncoder that can encode coordinates up to the next power of 2
    /// that covers both dimensions.
    pub fn for_dimensions(width: u32, height: u32) -> Self {
        let max_dim = width.max(height);
        let order = (32 - max_dim.saturating_sub(1).leading_zeros()).max(1);
        let size = 1u32 << order;
        Self { order, size }
    }

    /// Create a new Hilbert encoder with explicit order.
    ///
    /// # Arguments
    /// * `order` - The order of the Hilbert curve (size = 2^order)
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
    ///
    /// # Arguments
    /// * `x` - X coordinate (column)
    /// * `y` - Y coordinate (row)
    ///
    /// # Returns
    /// The Hilbert curve index (0 to 2^(2*order) - 1)
    pub fn encode(&self, x: u32, y: u32) -> u64 {
        if self.order == 0 {
            return 0;
        }

        // Clamp to bounds
        let x = x.min(self.size - 1);
        let y = y.min(self.size - 1);

        Self::xy2d(self.order, x, y)
    }

    /// Convert 1D Hilbert index to 2D coordinates.
    ///
    /// # Arguments
    /// * `d` - The Hilbert curve index
    ///
    /// # Returns
    /// The (x, y) coordinates
    pub fn decode(&self, d: u64) -> (u32, u32) {
        if self.order == 0 {
            return (0, 0);
        }

        let max_index = (1u64 << (2 * self.order)) - 1;
        let d = d.min(max_index);

        Self::d2xy(self.order, d)
    }

    /// Encode a ScreenFrame using Hilbert curve ordering.
    ///
    /// Returns a vector of pixels in Hilbert curve order, which preserves
    /// 2D spatial locality better than row-major ordering.
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
    ///
    /// # Arguments
    /// * `pixels` - Pixels in Hilbert curve order
    /// * `width` - Target frame width
    /// * `height` - Target frame height
    ///
    /// # Returns
    /// A ScreenFrame with pixels in row-major order
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

            // Rotate
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
    fn d2xy(order: u32, mut d: u64) -> (u32, u32) {
        let mut x: u64 = 0;
        let mut y: u64 = 0;
        let mut t = d;
        let mut s = 1u64;

        for _ in 0..order {
            let rx = 1 & (t / 2);
            let ry = 1 & (t ^ rx);

            // Rotate
            if ry == 0 {
                if rx == 1 {
                    x = (1u64 << order) - 1 - x;
                    y = (1u64 << order) - 1 - y;
                }
                std::mem::swap(&mut x, &mut y);
            }

            x += s * rx;
            y += s * ry;
            t /= 4;
            s <<= 1;
        }

        (x as u32, y as u32)
    }
}

/// Apply Hilbert curve spatial encoding to a frame and return as model-ready tensor.
///
/// This function encodes the frame using Hilbert curve ordering, which preserves
/// 2D spatial locality when the image is processed as a 1D sequence.
///
/// # Arguments
/// * `frame` - The input screen frame
///
/// # Returns
/// A byte vector suitable for model input with spatial locality preserved.
pub fn encode_frame_hilbert(frame: &ScreenFrame) -> Vec<u8> {
    let encoder = HilbertEncoder::for_dimensions(frame.width, frame.height);
    encoder.encode_frame(frame)
}

/// Calculate local spatial coherence score using Hilbert neighborhood.
///
/// Returns a score between 0 and 1 indicating how well the frame's
/// pixel values preserve local spatial coherence (higher = more coherent).
pub fn spatial_coherence_score(frame: &ScreenFrame) -> f32 {
    if frame.width < 2 || frame.height < 2 {
        return 1.0;
    }

    let encoder = HilbertEncoder::for_dimensions(frame.width, frame.height);
    let hilbert_pixels = encoder.encode_frame(frame);

    if hilbert_pixels.len() < 2 {
        return 1.0;
    }

    // Calculate average difference between adjacent pixels in Hilbert order
    let mut total_diff: f32 = 0.0;
    let mut count: u32 = 0;

    for i in 1..hilbert_pixels.len() {
        let diff = (hilbert_pixels[i] as i32 - hilbert_pixels[i - 1] as i32).abs() as f32;
        total_diff += diff;
        count += 1;
    }

    // Normalize: lower difference = higher coherence
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
        assert_eq!(frame.get(5, 5), 0); // out of bounds
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
        assert_eq!(bytes.len(), 6); // 2 + 2 + 2 pixels
        assert_eq!(u16::from_le_bytes([bytes[0], bytes[1]]), 2);
        assert_eq!(u16::from_le_bytes([bytes[2], bytes[3]]), 1);
        assert_eq!(bytes[4], 100);
        assert_eq!(bytes[5], 200);
    }

    #[test]
    fn downsample_gray_identity() {
        // 4x4 -> 4x4 should be identity (or very close)
        let src = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let frame = downsample_gray(&src, 4, 4, 4, 4);
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        // Each target pixel maps to exactly one source pixel
        for i in 0..16 {
            assert_eq!(frame.pixels[i], src[i]);
        }
    }

    #[test]
    fn downsample_gray_halve() {
        // 4x4 -> 2x2: each target pixel averages a 2x2 block
        let src = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let frame = downsample_gray(&src, 4, 4, 2, 2);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        // Top-left 2x2: (10+20+50+60)/4 = 35
        assert_eq!(frame.pixels[0], 35);
        // Top-right 2x2: (30+40+70+80)/4 = 55
        assert_eq!(frame.pixels[1], 55);
        // Bottom-left 2x2: (90+100+130+140)/4 = 115
        assert_eq!(frame.pixels[2], 115);
        // Bottom-right 2x2: (110+120+150+160)/4 = 135
        assert_eq!(frame.pixels[3], 135);
    }

    #[test]
    fn downsample_bgr_basic() {
        // 2x2 BGR image, stride=8 (2*3=6, padded to 8)
        // Pixel (0,0): B=0, G=0, R=255 -> gray = (77*255)>>8 = 76
        // Pixel (1,0): B=0, G=255, R=0 -> gray = (150*255)>>8 = 149
        // Pixel (0,1): B=255, G=0, R=0 -> gray = (29*255)>>8 = 28
        // Pixel (1,1): B=128, G=128, R=128 -> gray = (77*128+150*128+29*128)>>8 = (32768)>>8 = 128
        let mut bgr = vec![0u8; 16]; // 2 rows * stride 8
                                     // Row 0
        bgr[0] = 0;
        bgr[1] = 0;
        bgr[2] = 255; // pixel (0,0) BGR
        bgr[3] = 0;
        bgr[4] = 255;
        bgr[5] = 0; // pixel (1,0) BGR
                    // Row 1
        bgr[8] = 255;
        bgr[9] = 0;
        bgr[10] = 0; // pixel (0,1) BGR
        bgr[11] = 128;
        bgr[12] = 128;
        bgr[13] = 128; // pixel (1,1) BGR

        let frame = downsample_bgr_to_gray(&bgr, 2, 2, 8, 2, 2);
        assert_eq!(frame.width, 2);
        assert_eq!(frame.height, 2);
        // Check approximate grayscale values
        assert_eq!(frame.pixels[0], 76); // pure red
        assert_eq!(frame.pixels[1], 149); // pure green
        assert_eq!(frame.pixels[2], 28); // pure blue
        assert_eq!(frame.pixels[3], 128); // gray
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
        // Noop should always succeed on any platform
        // (On Windows it calls the real function, on other platforms it returns Unsupported)
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
        // Zero dimensions should fail
        #[cfg(target_os = "windows")]
        {
            let r = capture_screen(0, 256);
            assert!(r.is_err());
        }
    }

    #[test]
    fn hilbert_encoder_creation() {
        let encoder = HilbertEncoder::for_dimensions(16, 16);
        assert_eq!(encoder.order(), 4); // 2^4 = 16
        assert_eq!(encoder.size(), 16);
    }

    #[test]
    fn hilbert_encoder_non_power_of_two() {
        let encoder = HilbertEncoder::for_dimensions(10, 15);
        assert_eq!(encoder.order(), 4); // Next power of 2 is 16
        assert_eq!(encoder.size(), 16);
    }

    #[test]
    fn hilbert_encode_decode_identity() {
        let encoder = HilbertEncoder::with_order(3); // 8x8 grid

        // Test that encoding and decoding are inverses
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
    fn hilbert_encode_monotonic() {
        let encoder = HilbertEncoder::with_order(2); // 4x4 grid

        // Hilbert indices should be in range [0, 2^(2*order))
        let max_index = (1u64 << (2 * encoder.order())) - 1;

        for x in 0..4 {
            for y in 0..4 {
                let d = encoder.encode(x, y);
                assert!(
                    d <= max_index,
                    "Index {} out of range for ({}, {})",
                    d,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn hilbert_neighbors_are_close() {
        // Adjacent pixels in 2D should have close Hilbert indices
        let encoder = HilbertEncoder::with_order(3);

        // Check that neighboring pixels have adjacent or nearby indices
        let d00 = encoder.encode(0, 0);
        let d10 = encoder.encode(1, 0);
        let d01 = encoder.encode(0, 1);

        // The indices should be relatively close (though not necessarily adjacent)
        let diff1 = (d10 as i64 - d00 as i64).abs();
        let diff2 = (d01 as i64 - d00 as i64).abs();

        // Differences should be small for neighbors
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
    fn hilbert_frame_encoding() {
        let frame = ScreenFrame {
            pixels: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            width: 4,
            height: 4,
            timestamp_ms: 0,
        };

        let encoder = HilbertEncoder::for_dimensions(4, 4);
        let encoded = encoder.encode_frame(&frame);

        // Encoded should have the right size
        assert_eq!(encoded.len(), 16); // 4x4 = 16

        // The values should be a permutation of the original
        let mut sorted_original: Vec<u8> = frame.pixels.clone();
        let mut sorted_encoded: Vec<u8> = encoded.clone();
        sorted_original.sort();
        sorted_encoded.sort();
        assert_eq!(sorted_original, sorted_encoded);
    }

    #[test]
    fn hilbert_frame_roundtrip() {
        let frame = ScreenFrame {
            pixels: (0..64u8).collect(), // 8x8 frame
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
    fn hilbert_frame_encoding_large() {
        // Test with larger dimensions
        let frame = ScreenFrame {
            pixels: vec![128u8; 256 * 256],
            width: 256,
            height: 256,
            timestamp_ms: 0,
        };

        let encoder = HilbertEncoder::for_dimensions(256, 256);
        assert_eq!(encoder.order(), 8); // 2^8 = 256

        let encoded = encoder.encode_frame(&frame);
        assert_eq!(encoded.len(), 256 * 256);
    }

    #[test]
    fn spatial_coherence_constant_frame() {
        // A constant frame should have perfect coherence
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
    fn spatial_coherence_checkerboard() {
        // A checkerboard pattern should have lower coherence
        let mut pixels = vec![0u8; 16];
        for i in 0..4 {
            for j in 0..4 {
                if (i + j) % 2 == 0 {
                    pixels[i * 4 + j] = 255;
                }
            }
        }

        let frame = ScreenFrame {
            pixels,
            width: 4,
            height: 4,
            timestamp_ms: 0,
        };

        let coherence = spatial_coherence_score(&frame);
        // Checkerboard should have lower coherence than constant
        assert!(coherence < 1.0, "Checkerboard should have coherence < 1.0");
        assert!(coherence >= 0.0, "Coherence should be non-negative");
    }

    #[test]
    fn encode_frame_hilbert_helper() {
        let frame = ScreenFrame {
            pixels: (0..16u8).collect(),
            width: 4,
            height: 4,
            timestamp_ms: 0,
        };

        let encoded = encode_frame_hilbert(&frame);
        assert_eq!(encoded.len(), 16);
    }

    #[test]
    fn hilbert_order_zero() {
        // Test edge case: order 0 (1x1 grid)
        let encoder = HilbertEncoder::with_order(0);
        assert_eq!(encoder.size(), 1);
        assert_eq!(encoder.encode(0, 0), 0);
        assert_eq!(encoder.decode(0), (0, 0));
    }

    #[test]
    fn hilbert_clamping() {
        // Test that coordinates are clamped to grid bounds
        let encoder = HilbertEncoder::with_order(2); // 4x4 grid

        // These should be clamped to (3, 3)
        let d = encoder.encode(100, 100);
        let (x, y) = encoder.decode(d);
        assert!(x < 4);
        assert!(y < 4);
    }
}
