OBJ=eigenvalue_test
N=3
all:
	nvc++ main.cu -lcudart -lcublas -lcusolver -llapacke -llapack -lblas -use_fast_math -Xcompiler -fopenmp -lopenblas -O3 --diag_suppress set_but_not_used -o eigenvalue_test -DPSIZE=$(N)
run: all
	./$(OBJ)
sys: all
	nsys nvprof ./$(OBJ)