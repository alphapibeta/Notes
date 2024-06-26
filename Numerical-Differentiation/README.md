# Numerical Differentiation

This project implements numerical methods for differentiating mathematical functions. It includes forward, backward, and central differentiation methods of priliminary functions. 

## Project Structure

```
Numerical-Differentiation
├── docs
│   └── Numerical-differention.pdf
├── Makefile
├── README.md
└── src
    ├── docs
    ├── include
    │   ├── BackwardDerivative.h
    │   ├── CentralDerivative.h
    │   ├── ForwardDerivative.h
    │   └── MathFunctions.h
    ├── main.cpp
    ├── src
    │   ├── BackwardDerivative.cpp
    │   ├── CentralDerivative.cpp
    │   ├── ForwardDerivative.cpp
    │   └── MathFunctions.cpp
    └── tests
        └── MathFunctionsTest.cpp
```

## Building the Project

To compile the project, ensure you have `g++` and `make` installed on your system. Navigate to the project directory and run the following command:

```bash
make all
```


This will compile all source files and link them into an executable named derivative.

## Running the Application

After building the project, you can run the main application using:
``` bash
./derivative
```
[View Numerical Differentiation Documentation](docs/Numerical-differention.pdf)