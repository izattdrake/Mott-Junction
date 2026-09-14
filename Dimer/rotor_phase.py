import numpy as np
import matplotlib.pyplot as plt

from Decoupled.decoupled_dimer import DecoupledDimer
from Utility.plotting import save_axes_centered

if __name__ == "__main__":
    epsilon, t, beta, l_max = 0.0, 1.0, None, 15
    U_vals = np.array([0.04, 0.5, 1.0, 2.0, 3.0,3.9, 4.0, 4.01])

    theta_vals = np.linspace(0, 2*np.pi, 200)
    output_vals = np.zeros((len(theta_vals), len(U_vals)))

    count = 1

    for i in range(len(U_vals)):
        dc_dimer = DecoupledDimer(epsilon, U_vals[i], t, l_max, beta=None, print_cache_results=True)
        states = dc_dimer._evecs_rotor_vec[0]
        rotor_gs = states[:,0]
        output_vals[:,i] = dc_dimer.rotor_gs(theta_vals)

        print(count); count += 1

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
  
    for i in range(len(U_vals)):
        ax.plot(theta_vals, output_vals[:,i])
  
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$\Psi_0(\theta)$")
    ax.set_xticks(ticks=[0,np.pi,2*np.pi], labels=["0", r"$\pi$", r"$2\pi$"])

    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.legend(handlelength=2.2, labelspacing=0.35, borderaxespad=0.5)
  
    save_axes_centered(fig, ax, "Dimer/rotor_phase.pdf")
    plt.show()

