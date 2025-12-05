from fractions import Fraction
from math import sqrt
from fractions import Fraction
import sympy as sp
import ast

def run_simplex(points, function, reflexion=1, contraction=Fraction(1,2), expansion=2, epsilon=None, n_iter=4):
    def f(P):
        return function(P)
    
    # Convert points to Fractions
    points = [[Fraction(coord) for coord in P] for P in points]
    
    iteration = 0
    while True:
        # Calculate function values
        f_values = [f(P) for P in points]
        
        # Sort points according to function values
        # Tie-breaking: higher index gets larger function in case of equality
        sorted_indices = sorted(range(len(points)), key=lambda i: (f_values[i], -i))
        Ps = points[sorted_indices[0]]  # smallest
        Pnl = points[sorted_indices[1]] # second largest
        Pl = points[sorted_indices[2]]  # largest

        Fs = f(Ps)
        Fnl = f(Pnl)
        Fl = f(Pl)
        
        # Print current simplex
        print(f"\nIteration {iteration}:")
        print(f"Ps = {Ps}, Fs = {Fs}")
        print(f"Pnl = {Pnl}, Fnl = {Fnl}")
        print(f"Pl = {Pl}, Fl = {Fl}")
        
        # Calculate centroid Pg of Ps and Pnl
        Pg = [(Ps[i]+Pnl[i])/2 for i in range(len(Ps))]
        print(f"Pg = (Ps + Pnl)/2 = {Pg}")
        
        # Reflection
        Pr = [Pg[i] + reflexion*(Pg[i]-Pl[i]) for i in range(len(Pg))]
        Fr = f(Pr)
        # Print reflection step
        print(f"Pr = Pg + reflexion*(Pg - Pl) = {Pr}, Fr = {Fr}")
        print(f"Fr < Fs ? {'yes' if Fr < Fs else 'no'}")
        
        if Fr < Fs:
            # Expansion
            Pe = [Pg[i] + expansion*(Pr[i]-Pg[i]) for i in range(len(Pg))]
            Fe = f(Pe)
            print(f"Pe = Pg + expansion*(Pr - Pg) = {Pe}, Fe = {Fe}")
            if Fe < Fr:
                points[sorted_indices[2]] = Pe
                print(f"Expansion: replacing Pl with Pe")
            else:
                points[sorted_indices[2]] = Pr
                print(f"Reflection accepted: replacing Pl with Pr")
        elif Fr <= Fnl:
            points[sorted_indices[2]] = Pr
            print(f"Reflection accepted: replacing Pl with Pr")
        else:
            # Contraction
            if Fr < Fl:
                Pc = [Pg[i] + contraction*(Pr[i]-Pg[i]) for i in range(len(Pg))]
            else:
                Pc = [Pg[i] + contraction*(Pl[i]-Pg[i]) for i in range(len(Pg))]
            Fc = f(Pc)
            print(f"Pc = {Pc}, Fc = {Fc}")
            if Fc <= Fl:
                points[sorted_indices[2]] = Pc
                print(f"Contraction accepted: replacing Pl with Pc")
            else:
                # Shrink
                for i in range(1, len(points)):
                    points[i] = [Ps[j] + Fraction(1,2)*(points[i][j]-Ps[j]) for j in range(len(Ps))]
                print(f"Shrink (replace) performed")
        
        iteration += 1
        
        # Convergence check
        f_mean = sum(f(P) for P in points)/len(points)
        s = sqrt(sum((f(P)-f_mean)**2 for P in points)/(len(points)-1)) ## n instead of n+1 b/c lectures' notes says so...
        if epsilon is not None and s < epsilon:
            print(f"Epsilon stopping criterion reached: {s} < {epsilon}")
            print(f"f_mean: {f_mean}")
            print(f"s: {s}")
            break
        if n_iter is not None and iteration >= n_iter:
            print(f"Iteration stopping criterion reached: {iteration} >= {n_iter} ")
            print(f"f_mean: {f_mean}")
            print(f"s: {s}")
            break
        
    print("\nFinal simplex:")
    for i, P in enumerate(points):
        print(f"Point {i}: {P}, f = {f(P)}")
    # Final centroid
    Pg_final = [sum(points[i][j] for i in range(len(points)))/len(points) for j in range(len(points[0]))]
    Fg_final = f(Pg_final)
    print(f"\nFinal centroid (the average of ALL simplex points) Pg = {Pg_final}, F(Pg) = {Fg_final}")
    return Pg_final, Fg_final


func_str = input("Enter the function f(x,y) (e.g. 2*x^2 + 2*x*y + y^2 + x - y) = ")

# Create sympy expression
x, y = sp.symbols('x y')
func_expr = sp.sympify(func_str)

# Convert to Python callable
def user_function(P):
    subs = {x: P[0], y: P[1]}
    val = func_expr.evalf(subs=subs)
    # Try to convert to Fraction if possible
    try:
        return Fraction(val)
    except:
        return float(val)

# Ask for initial simplex points
points = []
print("Enter 3 simplex points, one per line (e.g., 1/2 0.5):")

for i in range(3):
    line = input(f"Point {i+1}: ")
    coords = line.split()
    point = [Fraction(coord) if '/' in coord else Fraction(str(float(coord))) for coord in coords]
    points.append(point)

print("Parsed points:", points)

# --- Ask for optional parameters ---
epsilon_input = input("Enter epsilon (leave blank (and press enter) if number of iterations is needed instead): ")
epsilon = float(epsilon_input) if epsilon_input else None

n_iter_input = input("Enter max iterations (leave blank for default 4): ")
n_iter = int(n_iter_input) if n_iter_input else None

reflexion_input = input("Enter reflection coefficient (leave blank for default 1): ")
reflexion = float(reflexion_input) if reflexion_input else 1

expansion_input = input("Enter expansion coefficient (leave blank for default 2): ")
expansion = float(expansion_input) if expansion_input else 2

contraction_input = input("Enter contraction coefficient (leave blank for default 1/2): ")
contraction = Fraction(contraction_input) if contraction_input else Fraction(1,2)



# --- Run simplex ---
Pg, Fg = run_simplex(points, user_function,
                     reflexion=reflexion,
                     expansion=expansion,
                     contraction=contraction,
                     epsilon=epsilon,
                     n_iter=n_iter)

print("\nOptimization result:")
print("Centroid Pg =", Pg)
print("Function at Pg =", Fg)


