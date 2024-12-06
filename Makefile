OBJ=eigenvalue_test
N=3
all:
	nvc++ main.cu -lcudart -lcublas -lcusolver -llapacke -llapack -lblas -use_fast_math -Xcompiler -fopenmp -lopenblas -O3 -o eigenvalue_test -DPSIZE=$(N) $(PRINT)
run: all
	./$(OBJ)
sys: all
	nsys nvprof ./$(OBJ)