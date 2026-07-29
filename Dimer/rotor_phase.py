import numpy as np
import matplotlib.pyplot as plt

from Decoupled.Decoupled_Dimer import DecoupledDimer

if __name__ == '__main__':
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15
    U_vals = np.array([0.04, 0.2, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.05, 4.1])

    theta_vals = np.linspace(0, 2*np.pi, 200)
    output_vals = np.zeros((len(theta_vals), len(U_vals)))

    count = 1

    for i in range(len(U_vals)):
        dc_dimer = DecoupledDimer(epsilon, U_vals[i], t, beta, l_max)
        output_vals[:,i] = dc_dimer.rotor_gs(theta_vals)

        print(count); count += 1
        
    plt.figure(figsize=(6, 4))
    for i in range(len(U_vals)):
        plt.plot(theta_vals, output_vals[:,i], label=f'U={U_vals[i]}')
    plt.xticks(ticks=[0,np.pi,2*np.pi], labels=['0', r'$\pi$', r'$2\pi$'])
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\Psi_0(\theta)$')
    plt.title('Rotor ground state')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
