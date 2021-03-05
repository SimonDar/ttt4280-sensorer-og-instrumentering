import matplotlib.pyplot as plt

# Plot bode
def plot_bode():
    filterx = []
    filtery = []
    f = open("csv/bode.csv")
    for line in f:
        temp = line.split(",")
        filterx.append(float(temp[0]))
        filtery.append(float(temp[1]))
    f.close()
    plt.plot(filterx, filtery, "b-")
    plt.title("Bodeplott")
    plt.xlabel("Frekvens [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.xscale("log")

plot_bode()
plt.savefig("plots/bode.png")
print("Bode plot saved in plots/bode.png.")