//! Checkpoint I/O: Save and load model/training state to disk.
//!
//! Implements a simple binary format for model checkpoints.

use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use sophon_config::{D_MODEL, NUM_BLOCKS, VOCAB_SIZE};
use sophon_model::Sophon1;

use crate::state::TrainState;

// Magic bytes: "SOPH1CKP" (Sophon-1 Checkpoint)
const MAGIC: &[u8] = b"SOPH1CKP";
const VERSION: u32 = 1;

/// Checkpoint format error.
#[derive(Debug)]
pub enum CheckpointError {
    Io(io::Error),
    InvalidMagic,
    InvalidVersion { expected: u32, found: u32 },
    CorruptedData(String),
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::InvalidMagic => write!(f, "invalid checkpoint magic bytes"),
            Self::InvalidVersion { expected, found } => {
                write!(
                    f,
                    "checkpoint version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::CorruptedData(msg) => write!(f, "corrupted checkpoint data: {}", msg),
        }
    }
}

impl std::error::Error for CheckpointError {}

impl From<io::Error> for CheckpointError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Save model and training state to a checkpoint file.
pub fn save_checkpoint<P: AsRef<Path>>(
    path: P,
    _model: &Sophon1,
    train_state: &TrainState,
) -> Result<(), CheckpointError> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(MAGIC)?;
    writer.write_all(&VERSION.to_le_bytes())?;

    // Write train state (minimal for now)
    writer.write_all(&train_state.global_step.to_le_bytes())?;
    writer.write_all(&train_state.ema_loss.to_le_bytes())?;

    // TODO: Write full model parameters
    // This would require exposing accessor methods for all model parameters

    writer.flush()?;
    Ok(())
}

/// Load model and training state from a checkpoint file.
pub fn load_checkpoint<P: AsRef<Path>>(
    path: P,
    _model: &mut Sophon1,
    train_state: &mut TrainState,
) -> Result<(), CheckpointError> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read and verify header
    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(CheckpointError::InvalidMagic);
    }

    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let version = u32::from_le_bytes(buf);
    if version != VERSION {
        return Err(CheckpointError::InvalidVersion {
            expected: VERSION,
            found: version,
        });
    }

    // Read train state
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf)?;
    train_state.global_step = u64::from_le_bytes(buf);

    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    train_state.ema_loss = f32::from_le_bytes(buf);

    // TODO: Read full model parameters

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sophon_model::Sophon1;

    #[test]
    fn checkpoint_roundtrip() {
        let model = Sophon1::new(42);
        let mut train_state = TrainState::new();
        train_state.global_step = 123;
        train_state.ema_loss = 2.5;

        let path = "test_checkpoint.ckpt";
        save_checkpoint(path, &model, &train_state).unwrap();

        let mut loaded_model = Sophon1::new(0);
        let mut loaded_state = TrainState::new();
        load_checkpoint(path, &mut loaded_model, &mut loaded_state).unwrap();

        assert_eq!(loaded_state.global_step, 123);
        assert!((loaded_state.ema_loss - 2.5).abs() < 1e-5);

        // Clean up
        std::fs::remove_file(path).unwrap();
    }
}
