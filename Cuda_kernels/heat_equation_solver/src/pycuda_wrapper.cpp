#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "HeatEquationSolver.h"

static PyObject* cpu_heat_solver(PyObject* self, PyObject* args) {
    int N, num_steps, num_threads;
    float alpha, dt, dx2, dy2;
    PyArrayObject* grid_obj;

    if (!PyArg_ParseTuple(args, "iiffffiO", &N, &num_steps, &alpha, &dt, &dx2, &dy2, &num_threads, &grid_obj)) {
        return NULL;
    }

    float* grid = (float*)PyArray_DATA(grid_obj);

    cpu_heat_equation_solver(N, num_steps, grid, alpha, dt, dx2, dy2, num_threads);

    Py_RETURN_NONE;
}

static PyObject* gpu_heat_solver(PyObject* self, PyObject* args) {
    int N, num_steps, block_thread_x, block_thread_y;
    float alpha, dt, dx2, dy2;
    PyArrayObject* grid_obj;

    if (!PyArg_ParseTuple(args, "iiffffiiO", &N, &num_steps, &alpha, &dt, &dx2, &dy2, &block_thread_x, &block_thread_y, &grid_obj)) {
        return NULL;
    }

    float* grid = (float*)PyArray_DATA(grid_obj);

    gpu_heat_equation_solver(N, num_steps, grid, alpha, dt, dx2, dy2, block_thread_x, block_thread_y);

    Py_RETURN_NONE;
}

static PyMethodDef HeatSolverMethods[] = {
    {"cpu_heat_solver", cpu_heat_solver, METH_VARARGS, "Solve heat equation using CPU"},
    {"gpu_heat_solver", gpu_heat_solver, METH_VARARGS, "Solve heat equation using GPU"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef heat_solver_module = {
    PyModuleDef_HEAD_INIT,
    "heat_solver",
    NULL,
    -1,
    HeatSolverMethods
};

PyMODINIT_FUNC PyInit_heat_solver(void) {
    import_array();
    return PyModule_Create(&heat_solver_module);
}