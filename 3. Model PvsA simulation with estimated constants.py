from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt


def model(t, variables, n, k1, k2, phi_in, phi_out, A_out, v, km, klanda, landa0):
    P, A = variables
    alpha = landa0/(1+klanda*A) #A initial o A a lo largo del tiempo(FUERA O DENTRO)
    dPdt = alpha*n*(1+k2)/((1+k1*A)**2*(1+k2*P)) - alpha*P
    dAdt = phi_in*A_out - phi_out*A - v*P*A/(km+P) - alpha*A
    return [dPdt, dAdt]


t_span = [0, 10]

#Estimated values ​​of the MEAN constants after the metropoli-hastings method
const_mean = [ 0.09350985, 0.58026128, 0.43635995, 0.54207802, -0.45358854, 0.6839959, 0.36064184, 0.65045037]
k1 ,k2, phi_in ,phi_out , v , km , klanda , alphaBeta = const_mean
initial_conditions = [2, 0]  

# Solve the ODE of two variables
sol1 = solve_ivp(model, t_span, initial_conditions, args=(n,k1,k2, phi_in, phi_out, 0, v, km, klanda, landa0), t_eval=np.linspace(0, 10, 100))
sol2 = solve_ivp(model, t_span, initial_conditions, args=(n,k1,k2, phi_in, phi_out, 1, v, km, klanda, landa0), t_eval=np.linspace(0, 10, 100))
sol3 = solve_ivp(model, t_span, initial_conditions, args=(n,k1,k2, phi_in, phi_out, 4, v, km, klanda, landa0), t_eval=np.linspace(0, 10, 100))
sol4 = solve_ivp(model, t_span, initial_conditions, args=(n,k1,k2, phi_in, phi_out, 10, v, km, klanda, landa0), t_eval=np.linspace(0, 10, 100))


plt.figure()


plt.subplot(2,2,1)
plt.plot(sol1.t, sol1.y[0], 'b-', label='P(t)')
plt.plot(sol1.t, sol1.y[1], 'r-', label='A(t)')
plt.xlabel('Time, Aout = 0')
plt.grid(True)

plt.subplot(2,2,2)
plt.plot(sol2.t, sol2.y[0], 'b-', label='P(t)')
plt.plot(sol2.t, sol2.y[1], 'r-', label='A(t)')
plt.xlabel('Time, Aout = 1')
plt.grid(True)

plt.subplot(2,2,3)
plt.plot(sol3.t, sol3.y[0], 'b-', label='P(t)')
plt.plot(sol3.t, sol3.y[1], 'r-', label='A(t)')
plt.xlabel('Time, Aout = 4')
plt.grid(True)

plt.subplot(2,2,4)
plt.plot(sol4.t, sol4.y[0], 'b-', label='P(t)')
plt.plot(sol4.t, sol4.y[1], 'r-', label='A(t)')
plt.xlabel('Time, Aout = 10')
plt.grid(True)


plt.suptitle('Protein and Antibiotics concentration along time')
plt.legend()




#2 For fixed n, how does P∗(last point//equilibrium) change as Aout increases (from 0 to 40)?

Aout_values = np.linspace(0, 40, 10)
Pstar_aout = np.zeros((10, 10)) #x=n//y = Aout

for n in range(1,11):
    for i in range(0,len(Aout_values)):
        sol = solve_ivp(model, t_span, initial_conditions, args=(n,k1,k2, phi_in, phi_out, Aout_values[i], v, km,klanda, landa0), t_eval=np.linspace(0, 10, 100))
        Pstar_aout[n-1,i] = sol.y[0][-1]
print(Pstar_aout)

plt.figure()
for i in range(1,Pstar_aout.shape[0]):
    plt.plot(Aout_values, Pstar_aout[i,], label='n = '+str(i))


plt.xlabel('A_out')
plt.ylabel('P*')
plt.title('Changes in P* as Aout increases')
plt.grid(True)
plt.legend()
#plt.show()


#3 For fixed Aout , how does P∗ increase as n increases (from 1 to 10)?

n_values = np.linspace(1, 11, 10)
Pstar_n = np.zeros((10, 10)) #x=n//y = Aout
Aout_values = np.linspace(0, 40, 10)

for i in range(0,len(Aout_values)):
    for j in range(0,len(n_values)):
        sol = solve_ivp(model, t_span, initial_conditions, args=(n_values[j],k1,k2, phi_in, phi_out, Aout_values[i], v, km, klanda, landa0), t_eval=np.linspace(0, 10, 100))
        Pstar_n[i,j] = sol.y[0][-1]
        
plt.figure()

for i in range(0,Pstar_n.shape[1]):
    plt.plot(n_values, Pstar_n[i,], label='Aout = ' + str(round(Aout_values[i])))


plt.xlabel('n')
plt.ylabel('P*')
plt.title('Changes in P* as n increases')
plt.grid(True)
plt.legend()
#plt.show()



#4 3D plot

n_values, Aout_values = np.meshgrid(n_values, Aout_values)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(n_values, Aout_values, Pstar_n, cmap='viridis')
ax.set_xlabel('Eje N')
ax.set_ylabel('Eje Aout')
ax.set_zlabel('Eje P*')
ax.set_title('3D plot')


plt.show()


