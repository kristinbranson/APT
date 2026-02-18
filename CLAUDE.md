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

## Matlab coding conventions
- Indents should all be two spaces.  Top-level functions should not be indented.  Never use tabs.
- Most identifiers should be lower camelcase.  Exceptions include: Class names, which should be upper camelcase.  The tags of UI controls should be snake case.  Methods of LabelerController that end in actuate_ should be snake case.
- Don't use private properties or methods in classes.  If a property or method would logically be private, add an underscore ("_") to the end of its name.  This signals to developers it is private "in spirit", but doesn't put in place annoying arbitrary restrictions when debugging.
- Public properties of classes should generally be Dependent.  If you're tempted to make a non-Dependent public-in-spirit property, make a private-in-spirit property that ends in an underscore, then make a Dependent property without the underscore.  Write get. and set. methods for the Dependent property as needed.
- Boolean variables should start with "tf" and then some conjugation of "to be".  For instance: tfIsDone, tfDidExplode, tfAreYouSure.
- Local variables in functions/methods should only be overwritten if necessary for performance.  Prefer to create new variables holding evolving versions of some value.
- Prefer explicit variable names, even if they are long; and avoid abbreviations.  Use a shorter English word that means the same thing instead of an abbreviation.
- Use spaces liberally in long expressions to add clarity.  E.g. add a space after each comma in the argument list for functions.
- But don't insert a space before the final semicolon in a line of code.
- Make properties that are not persisted to disk Transient.
- When checking for optional arguments, don't use nargin.  Use exist(<variable name>, 'var').  This is less likely to break when you add/remove arguments.
- The "end" keyword at the end of a function should be followed by the comment "% function".  Same for end of a methods block and a classdef block.
- Individual lines should not be longer than 160 characters.
- switch statements that check for multiple enumerated cases should enumerate all the handled cases explicitly, and throw an error in the "otherwise:" clause.  This makes it easier to find inappropriately-handled cases when testing.
- Classes should implement a char() method for producing a char array ("string") version of the object.
- I sometimes use the term "charray" for "char array".  OK to use this in comments, but just use "char" in variable names.
- When converting a custom class to a char array, write it as "char(thing)", not thing.char()

## Git conventions
- Always prepend the commit message with "<branch name>: ".  This
  makes it much easier to understand complicated git histories.
