# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
APT (Animal Part Tracker) is a machine-learning based software package for tracking animal pose/posture in video. It's a hybrid Matlab/Python codebase with a GUI frontend and deep learning backends.

## Architecture
- **Frontend**: Matlab GUI launched via `StartAPT()` in Matlab root directory
- **Backend**: Python deep learning implementations in `deepnet/` directory
- **Main interface**: `deepnet/APT_interface.py` - primary Python entry point for deep learning operations
- **Project format**: `.lbl` files containing movie paths and ground-truth labeling data

## Deep Learning Backends
APT supports 4 different backends for deep learning:
- **conda/docker**: Local machine training and evaluation
- **aws/bsub**: Remote machine training and evaluation on AWS or cluster

## Development Setup
### Matlab Setup
```matlab
modpath();  % Sets up Matlab path for APT
```

### Python Setup
Python components are primarily in `deepnet/` directory. The main interface is `APT_interface.py`.

## Testing
### Test Structure
- **Local backend tests**: `matlab/test/single-tests/` (conda, docker)
- **Remote backend tests**: `matlab/test/single-tests/remote/` (aws, bsub)
- Each test is a Matlab function that errors on failure, exits normally on success

### Running Tests
```matlab
test_apt()                    % Run all local backend tests
test_apt('remote', true)      % Run all tests including remote backends
test_apt('test_function_name') % Run single test for debugging
```

## Key Directories
- `matlab/`: Core Matlab codebase including GUI, algorithms, and tests
- `deepnet/`: Python deep learning implementations and training code
- `matlab/test/`: Test suite with local and remote test categories
- `docs/`: Documentation and user guides
- `examples/`: Example projects and data

## Important Files
- `StartAPT.m`: Main application entry point
- `modpath.m`: Path setup utility
- `deepnet/APT_interface.py`: Primary Python interface for deep learning
- `matlab/test/test_apt.m`: Main test runner
