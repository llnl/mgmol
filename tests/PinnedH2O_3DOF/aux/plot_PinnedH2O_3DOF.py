import matplotlib
import matplotlib.pyplot as plt
import numpy as np

ref_bondlength = 1.83
ref_bondangle = 104.5

# factors and increments for bond lengths and bond angle
bondlength_factor = np.linspace(0.95, 1.05, 3)
bondangle_increment = np.linspace(-5, 5, 3)

fig, ax = plt.subplots()

for d_bondangle in bondangle_increment:
    bondangle = ref_bondangle + d_bondangle
    x = ref_bondlength * np.cos(np.radians(bondangle / 2))
    y = ref_bondlength * np.sin(np.radians(bondangle / 2))
    for i, f_bondlength1 in enumerate(bondlength_factor):
        H1_x = f_bondlength1*x
        H1_y = f_bondlength1*y
        ax.plot(H1_x, H1_y, 'ro', markersize=8, alpha=0.1)
        for f_bondlength2 in bondlength_factor[:(i+1)]:
            H2_x = f_bondlength2*x
            H2_y = -f_bondlength2*y
            ax.plot(H2_x, H2_y, 'bo', markersize=8, alpha=0.1)

ref_bondangle = np.radians(ref_bondangle)
x = ref_bondlength * np.cos(ref_bondangle / 2)
y = ref_bondlength * np.sin(ref_bondangle / 2)

ax.plot([0, x], [0, y], 'r-', lw=2)
ax.plot([0, x], [0, -y], 'b-', lw=2)
ax.plot(0, 0, 'ko', markersize=8)
ax.plot(x, y, 'ro', markersize=8)
ax.plot(x, -y, 'bo', markersize=8)

arc1 = matplotlib.patches.Arc(xy=(0, 0), 
                             width=0.5,
                             height=0.5, 
                             angle=0, 
                             theta1=0, 
                             theta2=ref_bondangle/2,
                             color='red',
                             linewidth=2)
ax.add_patch(arc1)
ax.text(0.4, 0.25, r"$\frac{\theta}{2}$", ha='center', va='center', fontsize=12, color='red') 

arc2 = matplotlib.patches.Arc(xy=(0, 0), 
                             width=0.5,
                             height=0.5, 
                             angle=0, 
                             theta1=-ref_bondangle/2,
                             theta2=0, 
                             color='blue',
                             linewidth=2)
ax.add_patch(arc2)
ax.text(0.4, -0.25, r"$\frac{\theta}{2}$", ha='center', va='center', fontsize=12, color='blue') 

ax.text(x / 2 - 0.1, y / 2, r"$L_1$", ha='right', color='red')
ax.text(x / 2 - 0.1, -y / 2, r"$L_2$", ha='right', color='blue')

ax.plot([-2, 2], [0, 0], 'k--', lw=1)
ax.set_xlim(-2.0, 2.0)
ax.set_ylim(-2.0, 2.0)
ax.set_aspect('equal')
ax.set_xlabel("x (Bohr)")
ax.set_ylabel("y (Bohr)")
ax.grid(True)
plt.savefig("PinnedH2O_3DOF.png")
