import numpy as np
import matplotlib.pyplot as plt

class QuadraticSurface:
    def __init__(self, A, B, C, D, E, F, G, H, I, J):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.E = E
        self.F = F
        self.G = G
        self.H = H
        self.I = I
        self.J = J

        self.Mat_A = np.array(self.create_symmetric_matrix())
        self.Mat_B = self.get_vector_b()
        self.v1 = np.array([])
        self.v2 = np.array([])
        self.v3 = np.array([])
        self.w1 = np.array([])
        self.w2 = np.array([])
        self.w3 = np.array([])

    def create_symmetric_matrix(self):
        return [
            [self.A, self.D / 2, self.E / 2],
            [self.D / 2, self.B, self.F / 2],
            [self.E / 2, self.F / 2, self.C],
        ]

    def get_vector_b(self):
        return [self.G, self.H, self.I]

    def get_constant(self):
        return self.J

    def assign_columns_to_v(self):
        """Assign columns of Mat_A to w1, w2, and w3."""
        self.v1 = self.Mat_A[:, 0]  # First column
        self.v2 = self.Mat_A[:, 1]  # Second column
        self.v3 = self.Mat_A[:, 2]  # Third column

    def get_Q(self):
        self.Q = np.column_stack((self.w1_bar, self.w2_bar, self.w3_bar))
    
    def normalize(self, vector):   
        magnitude = self.get_magnitude(vector)
        if magnitude == 0:
            return vector 
        return vector / magnitude
    def get_magnitude(self, vector):
        return np.sqrt(np.sum(np.square(vector)))
    def QR_Factor(self, max_iter=900, tolerance=1e-6):
        self.assign_columns_to_v()
        
        # Perform Gram-Schmidt orthogonalization
        self.w1 = self.v1
        self.w1_bar = self.normalize(self.w1)

        self.w2 = np.subtract(
            self.v2,
            (
                np.dot(self.v2, self.w1) /
                np.dot(self.w1, self.w1)
            ) * self.w1
        )
        self.w2_bar = self.normalize(self.w2)
        self.w3 = np.subtract(
            self.v3,
            (
                np.dot(self.v3, self.w2) /
                np.dot(self.w2, self.w2)
            ) * self.w2
        ) - (
            np.dot(self.v3, self.w1) /
            np.dot(self.w1, self.w1)
        ) * self.w1
        self.w3_bar = self.normalize(self.w3)
        # Update the matrix with w1_bar,w2_bar,w3_bar to Q
        self.get_Q()
        self.R = np.round(np.matmul(np.transpose(self.Q), self.Mat_A), decimals=6)

        # New Mat_A for next iteration
        self.new_Mat_A = np.matmul(self.R, self.Q)
        self.new_Mat_A = np.where(np.abs(self.new_Mat_A) < 1e-5, 0, self.new_Mat_A)
        self.new_Mat_A = np.round(self.new_Mat_A, decimals=6)
        # Update Mat_A for the next iteration
        self.Mat_A = self.new_Mat_A

        # Call again if iterations are left
        if max_iter > 1:
            return self.QR_Factor(max_iter - 1, tolerance)
        
        # If max_iter reached
        return self.Q, self.R, self.Mat_A
    def get_shape(self, eigenvalues):
        # Check the eigenvalues to classify the surface
        pos, neg, zero = 0,0,0
        for val in eigenvalues:
            if val == 0:
                zero += 1
            elif val > 0:
                pos += 1
            else: neg += 1
        signiture = (pos, neg, zero)
        if signiture[0] == 3:
            return "Ellipsoid"
        elif signiture[0] == 2 and signiture[2] == 1:
            return "Elliptic Paraboloid"
        elif signiture[0] == 1 and signiture[1] == 1 and signiture[2] == 1:
            return "Hyperbolic paraboloid"
        elif signiture[0] == 2 and signiture[1] == 1:
            return "Hyperboloid of one sheet"

def get_eigenvals(factored_mat):
    eigens = []
    factored_mat.tolist()
    for i in range(len(factored_mat)):
        eigens.append(factored_mat[i][i])
    return eigens


# Complete QR factorization 3 Quadratics
quadratic1 = QuadraticSurface(A=2, B=0, C=13, D=4, E=7, F=6, G=2, H=3, I=7, J=3)
quadratic2 = QuadraticSurface(A=1, B=1, C=0, D=0, E=0, F=0, G=0, H=0, I=0, J=0)
quadratic3 = QuadraticSurface(A=1, B=1, C=1, D=0, E=0, F=0, G=0, H=0, I=0, J=-1)


# Get eigenvalues from diagonal after QR factorization
quadratic1.QR_Factor()
eigenvalues1 = get_eigenvals(quadratic1.new_Mat_A)

quadratic2.QR_Factor()
eigenvalues2 = get_eigenvals(quadratic2.new_Mat_A)

quadratic3.QR_Factor()
eigenvalues3 = get_eigenvals(quadratic3.new_Mat_A)



def generate_Z(shape, X, Y, a, b, c):
    if shape == 'Ellipsoid':
        Z2 = np.maximum(c**2 * (1 - X**2 / a**2 - Y**2 / b**2), 0)
        Z = np.sqrt(Z2)
    
    elif shape == 'Elliptic Paraboloid':
        Z = (X**2 / a**2 + Y**2 / b**2)
    
    elif shape == 'Hyperbolic Paraboloid':
        Z = (X**2 / a**2 ) - (Y**2 / b**2)
    
    elif shape == 'Hyperboloid of one sheet':
        Z2 = (c**2) * (X**2 / a**2 + Y**2 / b**2 - 1)
        Z2[Z2 < 0] = np.nan
        Z = np.sqrt(Z2)
    
    else:
        Z = np.full_like(X, np.nan)
    
    return Z


def create_plot(ax, quadratic, eigenvalues):
    # Plot the surface on the provided axes (ax)
    a, b, c = 2, 2, 3 
    shape = quadratic.get_shape(eigenvalues)
    x = np.linspace(-5, 5, 300)
    y = np.linspace(-5, 5, 300)
    X, Y = np.meshgrid(x, y)

    Z = generate_Z(shape, X, Y, a, b, c)

    # Plot the surfaces on the axis associated with the shape
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    ax.plot_surface(X, Y, -Z, cmap='viridis', alpha=0.7)

    # Set the title and eigenvalues 
    ax.set_title(shape)
    ax.text2D(0.5, -0.1, f"Eigen Values: {eigenvalues}", ha='center', fontsize=10, transform=ax.transAxes)

    # Set labels for each subplot
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_box_aspect([1, 1, 1])

# Create the figure and subplots 
fig = plt.figure(figsize=(18, 6))  

# Create subplots 
ax1 = fig.add_subplot(131, projection='3d')  
ax2 = fig.add_subplot(132, projection='3d')  
ax3 = fig.add_subplot(133, projection='3d')  

create_plot(ax1, quadratic1, eigenvalues1)
create_plot(ax2, quadratic2, eigenvalues2)
create_plot(ax3, quadratic3, eigenvalues3)

# Adjust layout and show the plot
plt.subplots_adjust(bottom=0.15, wspace=0.3) 
plt.show()

# Print results 
def print_results(quadratic, eigenvalues):
    print("Final QR factorization Matrix:")
    print(quadratic.new_Mat_A)
    print(f"Matrix B {quadratic.get_vector_b()}^t")
    print("Eigen values:", eigenvalues)
    print(quadratic.get_shape(eigenvalues))



print("Quadratic 1:")
print_results(quadratic1, eigenvalues1)
print("Quadratic 2:")
print_results(quadratic2, eigenvalues2)
print("Quadratic 3:")
print_results(quadratic3, eigenvalues3)