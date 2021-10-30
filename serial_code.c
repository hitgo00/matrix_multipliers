init(N):
	Return new matrix of size N*N.

add(A,B,N):
	Returns addition of matrices A,B of size = N.

subtract(A,B,N):
	Returns subtraction of matrices A,B of size = N.

strassen(A,B,N):
	//resultant matrix of the current recursive call.
	C = init(N)
	
	//Divide
	K = N/2
	
	//Make new matrices.

	A11 = init(K)
	A12 = init(K)
	A21 = init(K)
	A22 = init(K)

	B11 = init(K)
	B12 = init(K)
	B21 = init(K)
	B22 = init(K)

	//Divide matrices A and B into 4 parts and fill them.

	i = [0….K-1]:
          	     j = [0…..K-1]:
		A11[i][j] = A[i][j]
		A12[i][j] = A[i][k+j]
		A21[i][j] = A[i+k][j]
		A22[i][j] = A[i+k][j+k]
		
		B11[i][j] = B[i][j]
		B12[i][j] = B[i][k+j]
		B21[i][j] = B[i+k][j]
		B22[i][j] = B[i+k][j+k]
	
	
	//Apply strassen’s equations.

	P1 = strassen(A11, subtract(B12, B22, k), k);
	P2 = strassen(add(A11, A12, k), B22, k);
	P3 = strassen(add(A21, A22, k), B11, k);
	P4 = strassen(A22, subtract(B21, B11, k), k);
	P5 = strassen(add(A11, A22, k), add(B11, B22, k), k);
	P6 = strassen(subtract(A12, A22, k), add(B21, B22, k), k);
	P7 = strassen(subtract(A11, A21, k), add(B11, B12, k), k);

	//Reorder  p1,p2,p3,p4,p5,p6,p7 to get c11,c12,c21,c22.
	//Conquer
	C11 = subtract(add(add(P5, P4, k), P6, k), P2, k);
	C12 = add(P1, P2, k)
  	C21 = add(P3, P4, k)
	C22 = subtract(subtract(add(P5, P1, k), P3, k), P7, k);

		
	//Remap C11,C12,C21,C22 to C and return it.

	i=[0….k-1]:
	   j=[0…..k-1]:
		C[i][j] = C11[i][j]
		C[i][j+k] = C12[i][j]
		C[i+k][j] = C21[i][j]
		C[i+k][j+k] = C22[i][j]

	//Cleanup used matrices expect A,B,C
	free(everything except A,B,C)
	
	
	//Return result
	return C

