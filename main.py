#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from abc import ABC, abstractmethod
import pickle
from pathlib import Path
import os
import multiprocessing
import time as systime
from typing import Tuple
import copy

plt.rcParams['figure.figsize'] = [15, 3]


class Thermostat(ABC):
    @abstractmethod
    def get_velocities(self, velocities, num_particles, dt, assigned_temperature, thermostat_strength):
        pass

    @abstractmethod
    def get_thermostat_name(self) -> str:
        pass


class Berendsen(Thermostat):
    def get_velocities(self, velocities, num_particles, dt, assigned_temperature, thermostat_strength):
        tau = thermostat_strength
        T = np.sum(velocities**2)/(num_particles*3)
        lamb = np.sqrt((1 + dt/tau * (assigned_temperature/T - 1)))
        velocities = lamb*velocities

        return velocities

    def get_thermostat_name(self) -> str:
        return 'Berendsen'


class NoThermostat(Thermostat):
    def get_velocities(self, velocities, num_particles, dt, assigned_temperature, thermostat_strength):
        return velocities

    def get_thermostat_name(self) -> str:
        return 'NoThermostat'


class Andersen(Thermostat):
    def get_velocities(self, velocities, num_particles, dt, assigned_temperature, thermostat_strength):
        nu, sigma = thermostat_strength, np.sqrt(assigned_temperature)
        collision_prob = nu*dt

        for i in range(num_particles):
            p = np.random.random()

            if p <= collision_prob:
                velocities[i] = np.random.normal(0, sigma, (1, 3))

        return velocities

    def get_thermostat_name(self) -> str:
        return 'Andersen'


class DormentThermostat(Thermostat):
    def __init__(self, activate_at: int, thermostat: Thermostat):
        self.iterations = 0
        self.activate_at = activate_at
        self.thermostat = thermostat

        if isinstance(thermostat, DormentThermostat):
            raise Exception(
                "Using DormentThermostat as an argument here will cause an infinite loop.")

    def get_velocities(self, velocities, num_particles, dt, assigned_temperature, thermostat_strength):
        updated_velocities = velocities

        if self.iterations >= self.activate_at:
            updated_velocities = self.thermostat.get_velocities(
                velocities, num_particles, dt, assigned_temperature, thermostat_strength)

        self.iterations += 1

        return updated_velocities

    def get_thermostat_name(self) -> str:
        return self.thermostat.__class__.__name__


class Barostat(ABC):
    @abstractmethod
    def get_positions(self, positions, h, strength, current_pressure, target_pressure):
        pass


class BerendsenBarostat(Barostat):
    def __init__(self, strength: float, target_pressure: float, activate_at: int):
        self.strength = strength
        self.target_pressure = target_pressure

        self.iterations = 0
        self.activate_at = activate_at

    def get_positions(self, positions, h, current_pressure):
        self.iterations += 1

        if self.iterations > self.activate_at:
            return positions * np.cbrt(1 + ((h/self.strength) * (current_pressure - self.target_pressure)))

        return positions


class NoBarostat(Barostat):
    def get_positions(self, positions, h, current_pressure):
        return positions


class GraphOptions():
    def __init__(self):
        self.title = 'No title'
        self.axis_labels = ['X axis', 'Y axis']
        self.x_values = []
        self.y_values = []
        self.legend = ['No label']


def show_graph(options: GraphOptions):
    plt.Figure()
    plt.title(options.title)

    for i, y in enumerate(options.y_values):
        plt.plot(options.x_values, y, label=options.legend[i])

    plt.xlabel(options.axis_labels[0])
    plt.ylabel(options.axis_labels[1])
    plt.legend()
    plt.show()


class SimulationOptions:
    def __init__(self):
        self.n: int = 50
        self.initial_temp: float = 5
        self.target_temp: float = self.initial_temp
        self.box_size: int = 15
        self.h: float = 0.01
        self.iterations: int = 1000
        self.thermostat_strength: int = 0
        self.thermostat: Thermostat = Andersen()
        self.barostat: Barostat = NoBarostat()

    def time_max(self) -> float:
        return self.iterations * self.h

    def __str__(self) -> str:
        return (
            f'{self.n}_{self.initial_temp}_{self.target_temp}_'
            f'{self.box_size}_{self.h}_{self.iterations}_{self.thermostat_strength}_'
            f'{self.thermostat.get_thermostat_name()}'
            f'{self.barostat.__class__.__name__}'
        )


def initialize(num_particles, temp, box_size, dt):
    positions = np.random.rand(num_particles, 3) * box_size
    velocities = np.random.standard_normal(size=(num_particles, 3))

    COM_velocities = np.sum(velocities, axis=0) / num_particles
    COM_velocities2 = np.sum(velocities**2, axis=0) / num_particles
    scaling_factor = np.sqrt(temp / COM_velocities2)

    velocities = (velocities - COM_velocities) * scaling_factor

    sum_velocities2 = np.sum(velocities**2)
    kinetic_energy = sum_velocities2/2

    return positions, velocities, kinetic_energy


def force_function(positions, box_size, kin_energy=None, cut_off_distance=3.5):
    cut_off_distance_2 = cut_off_distance**2

    num_particles = len(positions)
    forces = np.zeros((num_particles, 3), float)
    pressure_values = np.zeros(num_particles, float)
    pot_energy = 0.

    for i in range(num_particles - 1):
        rf = []

        for j in range(i + 1, num_particles):
            distance = positions[i] - positions[j]

            for dim in range(3):
                if abs(distance[dim]) > 0.5 * box_size:
                    distance[dim] = distance[dim] - \
                        np.sign(distance[dim]) * box_size

            r = np.sqrt(np.sum(distance**2))
            r2 = r**2

            if r2 <= cut_off_distance_2:
                # Stability improvement.
                if r2 < 0.001:
                    r2 = 0.001

                r2i = 1 / r2
                r6i = r2i**3

                rc2i = 1 / (cut_off_distance_2)
                rc6i = rc2i**3

                ff = 48*r2i*r6i*(r6i - 0.5)

                forces[i] += ff*distance
                forces[j] -= ff*distance

                pot_energy += 4*r6i*(r6i - 1) - 4*(rc6i**2 - rc6i)

                rf.append(r * forces[i])

        if kin_energy is not None:
            w: float = np.sum(rf)
            density: float = num_particles / box_size**3
            pressure_values[i] = (
                density / (3 * num_particles)) * (2 * kin_energy + w)

    return forces, pot_energy, np.sum(pressure_values)


def call_thermostat(velocities, num_particles, dt, thermostat: Thermostat, assigned_temperature, thermostat_strength):
    return thermostat.get_velocities(velocities, num_particles, dt, assigned_temperature, thermostat_strength)


def integrate(positions, velocities, forces, dt, box_size, num_particles,
              thermostat_choice, assigned_temperature, thermostat_strength, kin_energy, barostat_choice: Barostat):
    num_particles = len(positions)

    positions = positions + velocities*dt + 0.5*dt**2*forces

    # Correct for particles leaving the box. Using np.where, we find the coordi-
    # nates of the particles that are outside of the box after time dt.
    # Then, we correct the position using the indices returned by np.where.
    for dim in range(3):
        # update the positions by subtracting the box_size IF the coordinate is
        # larger than the box size.
        out_of_box = np.where(positions[:, dim] > box_size)
        positions[out_of_box, dim] = positions[out_of_box, dim] - box_size

        # repeat for other side
        out_of_box = np.where(positions[:, dim] < 0.)
        positions[out_of_box, dim] = positions[out_of_box, dim] + box_size

    # A bit of a dirty hack, but hey, it's 8pm on the day before the deadline and I'm tired.
    cut_off_distance: float = 3.5 if barostat_choice.__class__.__name__ == 'NoBarostat' else 0.5 * box_size

    new_forces, potential_energy, pressure = force_function(positions, box_size, kin_energy, cut_off_distance)
    velocities += 0.5*(forces+new_forces)*dt

    velocities = call_thermostat(velocities, num_particles, dt,
                                 thermostat_choice, assigned_temperature, thermostat_strength)

    positions = barostat_choice.get_positions(positions, dt, pressure)

    kin_energy = 0.5*np.sum(velocities**2)

    return positions, velocities, kin_energy, new_forces, potential_energy, pressure


def run_MD_simulation(options: SimulationOptions):
    particle_density: float = options.n/(options.box_size**3)

    print(':: Initializing MD simulation..')
    print('=> Initial particle density = ', particle_density)

    # A bit of a dirty hack, but hey, it's 8pm on the day before the deadline and I'm tired.
    cut_off_distance: float = 3.5 if options.barostat.__class__.__name__ == 'NoBarostat' else 0.5 * options.box_size

    positions, velocities, kin_energy_initial = initialize(options.n, options.initial_temp, options.box_size, options.h)
    forces, pot_energy_initial, pressure_initial = force_function(
        positions, options.box_size, cut_off_distance=cut_off_distance)

    # instantaneous normalization for the initial values
    kin_energy_initial /= options.n
    pot_energy_initial /= options.n
    temp_initial = np.sum(velocities**2)/(3*options.n)
    time = 0.

    time_array = np.array([time, ])
    pot_energy_array = np.array([pot_energy_initial, ])
    kin_energy_array = np.array([kin_energy_initial, ])
    temp_array = np.array([temp_initial, ])
    tot_energy_array = np.array([kin_energy_initial+pot_energy_initial, ])

    coordinate_array = [positions]
    velocities_array = [velocities]
    pressure_array = [pressure_initial]

    last_kin_energy = kin_energy_array[0]
    while time < options.time_max():
        positions, velocities, kin_energy, forces, pot_energy, pressure = integrate(
            positions, velocities, forces, options.h, options.box_size, options.n, options.thermostat, options.target_temp,
            options.thermostat_strength, last_kin_energy, options.barostat)
        time += options.h

        time_array = np.append(time_array, time)
        pot_energy_array = np.append(pot_energy_array, pot_energy/options.n)
        kin_energy_array = np.append(kin_energy_array, kin_energy/options.n)
        temp_array = np.append(temp_array, 2*kin_energy/(3*options.n))
        tot_energy_array = np.append(tot_energy_array, (kin_energy+pot_energy)/options.n)

        pressure_array.append(pressure)
        coordinate_array.append(positions)
        velocities_array.append(velocities)
        last_kin_energy = kin_energy.copy()

    return time_array, tot_energy_array, pot_energy_array, kin_energy_array, temp_array, coordinate_array, velocities_array, pressure_array


def draw_energy_plot(time, energy_tot, energy_kin, energy_pot, temp):
    graphOptions = GraphOptions()

    graphOptions.title = 'Energy vs time'
    graphOptions.axis_labels = ['Time', 'Energy in a.u.']
    graphOptions.x_values = time
    graphOptions.y_values = [energy_tot, energy_kin, energy_pot]
    graphOptions.legend = ['total energy', 'kinetic energy', 'potential energy']

    show_graph(graphOptions)

    graphOptions.title = 'Temperature vs time'
    graphOptions.y_values = [temp]
    graphOptions.axis_labels = ['Time', 'Temperature in a.u.']
    graphOptions.legend = ['Temperature']
    show_graph(graphOptions)

    nmax = len(energy_tot)
    drift = np.array([(1/nmax) * np.sum([np.abs((energy_tot[0] - energy_tot[i]) /
                                                energy_tot[0]) for i in range(0, t)]) for t in range(len(time))])

    graphOptions.title = 'Energy drift vs time'
    graphOptions.y_values = [drift]
    graphOptions.legend = ['Energy drift']
    graphOptions.axis_labels = ['Time', 'Energy drift in a.u.']
    show_graph(graphOptions)


def update(t, graph, coordinates):
    graph._offsets3d = (coordinates[t][:, 0], coordinates[t][:, 1], coordinates[t][:, 2])

    return graph,


def draw_md_animation(coordinates):
    r = coordinates[0]
    fig = plt.figure(dpi=100)
    ax = Axes3D(fig)
    graph = ax.scatter(r[:, 0], r[:, 1], r[:, 2], s=80)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    ani = animation.FuncAnimation(fig, lambda t: update(t, graph, coordinates),)
    plt.show()

    ani.save('MD_animation.gif', writer='pillow')


def run_md_simulation_with_timestep(h: float, iterations: int = 10000):
    options = SimulationOptions()
    options.h = h
    options.iterations = iterations

    start_time = systime.time()
    time, energy_tot, energy_pot, energy_kin, temp, coordinates, _, _ = run_MD_simulation(options)

    time_elapsed = systime.time() - start_time
    print(f'Real time: {time_elapsed} seconds --- Performance: {len(time) / time_elapsed} time steps per second')

    draw_md_animation(coordinates)
    draw_energy_plot(time, energy_tot, energy_kin, energy_pot, temp)


def run_md_simulation_with_temperature(temperature: float, h: int = 0.01, iterations: int = 10000):
    options = SimulationOptions()
    options.h = h
    options.initial_temp = temperature
    options.iterations = iterations

    start_time = systime.time()
    time, energy_tot, energy_pot, energy_kin, temp, coordinates, _, _ = run_MD_simulation(options)

    time_elapsed = systime.time() - start_time
    print(f'Real time: {time_elapsed} seconds --- Performance: {len(time) / time_elapsed} time steps per second')

    draw_md_animation(coordinates)
    draw_energy_plot(time, energy_tot, energy_kin, energy_pot, temp)


def run_md_simulation_with_various_n(n_values: int, h: int = 0.01, iterations: int = 10000):
    energy_totals = []
    energy_potentials = []
    time_values = None

    options = SimulationOptions()
    options.h = h
    options.iterations = iterations

    for n in n_values:
        options.n = n
        start_time = systime.time()

        time, energy_tot, energy_pot, _, _, _, _, _ = run_MD_simulation(options)

        energy_totals.append(energy_tot)
        energy_potentials.append(energy_pot)

        if time_values is None:
            time_values = time

        time_elapsed = systime.time() - start_time
        print(f'n={n} --- Real time: {time_elapsed} seconds --- Performance: {len(time) / time_elapsed} time steps per second')

    graphOptions = GraphOptions()

    graphOptions.title = 'Total energy vs time'
    graphOptions.axis_labels = ['Time', 'Total energy in a.u.']
    graphOptions.legend = ['n = {n}' for n in n_values]
    graphOptions.x_values = time_values
    graphOptions.y_values = energy_totals
    show_graph(graphOptions)

    graphOptions.title = 'Potential energy vs time'
    graphOptions.axis_labels = ['Time', 'Potential energy in a.u.']
    graphOptions.y_values = energy_potentials
    show_graph(graphOptions)


def run_coupling_analyses(h: int = 0.01):
    thermostat_strengths = np.linspace(0, 50, 20000)[1:]

    graphOptions = GraphOptions()

    graphOptions.title = 'Coupling vs thermostat strength'
    graphOptions.axis_labels = ['Thermostat strength', 'Coupling constant']
    graphOptions.legend = ['Andersen thermostat', 'Berendsen thermostaat']
    graphOptions.x_values = thermostat_strengths
    graphOptions.y_values = [[h*t for t in thermostat_strengths], [h/t for t in thermostat_strengths]]
    show_graph(graphOptions)


def run_with_dorment_thermostat(options: SimulationOptions):
    thermostat_name: str = options.thermostat.__class__.__name__
    options.thermostat = DormentThermostat(options.iterations // 2, options.thermostat)

    filename: str = f'Data/{options}'

    if Path(filename).is_file():
        print(f'Simulation has already been run: {options}')

        return

    time, energy_tot, energy_pot, energy_kin, temp, coordinates, velocities, _ = run_MD_simulation(options)

    with open(filename, 'wb') as file:
        pickle.dump({
            'thermostat': thermostat_name,
            'thermostat_strength': options.thermostat_strength,
            'time': time,
            'energy_tot': energy_tot,
            'energy_pot': energy_pot,
            'energy_kin': energy_kin,
            'temp': temp,
            'coordinates': coordinates,
            'velocities': velocities
        }, file)

    print(f'Simulation completed: {options}')


def retrieve_and_plot_simulation_data():
    if not Path('Data').is_dir():
        print('Cannot read from `Data` directory, as it does not exist.')

        return

    files: Array[str] = []
    for filename in os.listdir('Data'):
        path: string = os.path.join('Data', filename)

        if Path(path).is_file():
            files.append(path)

    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=100)
    fig.set_size_inches(10, 6)
    plt.tight_layout()

    for filename in files:
        with open(filename, 'rb') as results:
            result = pickle.load(results)
            axis: int = 0 if result['thermostat'] == 'Andersen' else 1

            axes[axis].plot(result['time'], result['temp'],
                            label=f'{result["thermostat"]} at strength={result["thermostat_strength"]}')

            print(f'Loaded file: {filename}')

    for axis in axes:
        axis.legend()
        axis.set_xlabel('Time')
        axis.set_ylabel('Temperature')

    plt.show()


def run_thermostat_strengths_test():
    def run_test(strength: int, thermostat: Thermostat):
        options = SimulationOptions()
        options.n = 50
        options.target_temp = 2.5
        options.h = 0.01
        options.iterations = 100 / options.h
        options.thermostat = thermostat
        options.thermostat_strength = strength

        run_with_dorment_thermostat(options)

    # Applying multiprocessing since patience is a virtue I do not possess. :P
    processes = []
    for strength in [50, 5, 0.5, 0.05]:
        processes.append(multiprocessing.Process(target=run_test, args=(strength, Andersen())))
        processes.append(multiprocessing.Process(target=run_test, args=(strength, Berendsen())))

    for process in processes:
        process.start()
        systime.sleep(2)

    for process in processes:
        process.join()


def plot_msd(time, labels, msd_values, strength: float):
    plt.Figure()

    if len(labels) != len(msd_values):
        print('Length mismatch for labels and MSD arrays')

        return

    plt.xlabel('Time')
    plt.ylabel('MSD')
    plt.title(f'MSD vs time at strength={strength}')

    for i in range(0, len(labels)):
        plt.plot([t for t, v in msd_values[i]], [
                 v for t, v in msd_values[i]], label=f'{labels[i]}')

    plt.legend()
    plt.show()


def mean_square_displacement_analysis(options: SimulationOptions):
    labels = ['Andersen thermostat', 'Berendsen thermostat']

    def run(strength: float, options: SimulationOptions):
        msd_values = []

        # Thermostat enables at iterations // 2. We want the index of the iteration preceding
        # this point. Since arrays start at 0, we will need to subtract 2 to get the index.
        start_at: int = (options.iterations // 2) - 2
        options.thermostat = DormentThermostat(options.iterations // 2, Andersen())
        options.thermostat_strength = strength

        time, _, _, _, _, coordinates, velocities, _ = run_MD_simulation(options)
        msd_values.append(calculate_msd(coordinates[start_at:], time[start_at:]))
        perform_distribution_comparison(velocities[start_at + 1:], 'Andersen', strength)

        options.thermostat = DormentThermostat(options.iterations // 2, Berendsen())
        time, _, _, _, _, coordinates, velocities, _ = run_MD_simulation(options)
        msd_values.append(calculate_msd(coordinates[start_at:], time[start_at:]))
        perform_distribution_comparison(velocities[start_at + 1:], 'Berendsen', strength)

        plot_msd(time, labels, msd_values, strength)

    processes = []
    for strength in [5000, 500, 50, 5, 0.5, 0.05]:
        processes.append(multiprocessing.Process(target=run, args=(strength, copy.deepcopy(options))))

    for process in processes:
        process.start()
        systime.sleep(2)

    for process in processes:
        process.join()


def calculate_msd(positions, times):
    n: int = len(positions[0])

    mean_squared_displacements: Array[Tuple[float, float]] = []

    for t in range(1, len(times)):
        current_positions = positions[t]
        initial_positions = positions[0]

        dist_x = (np.sum(current_positions[:, 0] - initial_positions[:, 0])) ** 2
        dist_y = (np.sum(current_positions[:, 1] - initial_positions[:, 1])) ** 2
        dist_z = (np.sum(current_positions[:, 2] - initial_positions[:, 2])) ** 2

        mean_squared_displacements.append((times[t], (np.sum(dist_x + dist_y + dist_z)) / n))

    return mean_squared_displacements


def perform_distribution_comparison(velocities, thermostat: str, strength: float):
    flattened_velocities = np.array(velocities).flatten()

    plt.figure()
    plt.hist(flattened_velocities, 50, density=True, color='b')

    mu, std = scipy.stats.norm.fit(flattened_velocities)

    xmin, xmax = plt.xlim()
    x_values = np.linspace(xmin, xmax, 50)
    plt.plot(x_values, scipy.stats.norm.pdf(x_values, mu, std), linewidth=2, color='r')
    plt.title(f'Velocity distribution of {thermostat} at strength={strength}')

    plt.show()


def plot_pressure_vs_time(options: SimulationOptions):
    pressure_values = []

    for thermostat in [Andersen(), Berendsen()]:
        options.thermostat = DormentThermostat(options.iterations // 2, thermostat)

        time, _, _, _, _, _, _, pressure = run_MD_simulation(options)

        pressure_values.append((thermostat.get_thermostat_name(), pressure))

    graphOptions = GraphOptions()

    graphOptions.title = 'Pressure vs time'
    graphOptions.axis_labels = ['Time', 'Pressure in a.u.']
    graphOptions.x_values = time
    graphOptions.y_values = [result[1] for result in pressure_values]
    graphOptions.legend = ['Andersen', 'Berendsen']

    show_graph(graphOptions)


def density_effect_analysis(options: SimulationOptions):
    #densities: Array[float] = [0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0, 1.15, 1.30, 1.45]
    densities: Array[float] = [0.015, 0.03, 0.045, 0.06]
    n_values = [int((options.box_size**3) * rho) for rho in densities]

    pressure_values = [[] for i in range(len(densities))]
    temperature_values = [[] for i in range(len(densities))]

    for i, n in enumerate(n_values):
        options.n = n
        options.thermostat = Andersen()

        time, _, _, _, temp, _, _, pressure = run_MD_simulation(options)

        pressure_values[i] = pressure
        temperature_values[i] = temp

    graphOptions = GraphOptions()

    graphOptions.title = 'Pressure vs time'
    graphOptions.axis_labels = ['Time', 'Pressure in a.u.']
    graphOptions.x_values = time
    graphOptions.y_values = pressure_values
    graphOptions.legend = [f'Density={rho}' for rho in densities]

    show_graph(graphOptions)

    graphOptions.title = 'Temperature vs time'
    graphOptions.axis_labels = ['Time', 'Temperature in a.u.']
    graphOptions.y_values = temperature_values

    show_graph(graphOptions)


def run_barostat_strengths_test():
    strengths = [50, 500, 5000]
    pressure_values = [[] for i in range(len(strengths))]
    temperature_values = [[] for i in range(len(strengths))]

    options = SimulationOptions()
    options.thermostat = NoThermostat()

    for i, strength in enumerate(strengths):
        options.barostat = BerendsenBarostat(strength, 1.0, options.iterations // 2)

        time, _, _, _, temp, _, _, pressure = run_MD_simulation(options)

        pressure_values[i] = pressure
        temperature_values[i] = temp

    graphOptions = GraphOptions()

    graphOptions.title = 'Pressure vs time'
    graphOptions.axis_labels = ['Time', 'Pressure in a.u.']
    graphOptions.legend = [f'Barostat strength = {strength}' for strength in strengths]
    graphOptions.x_values = time
    graphOptions.y_values = pressure_values
    show_graph(graphOptions)

    graphOptions.title = 'Temperature vs time'
    graphOptions.axis_labels = ['Time', 'Temperature in a.u.']
    graphOptions.y_values = temperature_values
    show_graph(graphOptions)


run_md_simulation_with_timestep(0.0001, iterations=10000)
run_md_simulation_with_timestep(0.001, iterations=10000)
run_md_simulation_with_timestep(0.005, iterations=2000)
run_md_simulation_with_timestep(0.01, iterations=1000)
run_md_simulation_with_timestep(0.05, iterations=200)
run_md_simulation_with_timestep(0.1, iterations=100)

run_md_simulation_with_temperature(5)
run_md_simulation_with_temperature(7)
run_md_simulation_with_temperature(9)
run_md_simulation_with_temperature(11)

run_md_simulation_with_various_n([50, 75, 100, 125])

run_thermostat_strengths_test()
retrieve_and_plot_simulation_data()

options = SimulationOptions()
options.iterations = 10000
mean_square_displacement_analysis(options)

options = SimulationOptions()
options.n = 50
options.target_temp = 2.5
options.h = 0.01
options.iterations = 100 / options.h
options.thermostat = Andersen()
options.thermostat_strength = 0.1
plot_pressure_vs_time(options)

options = SimulationOptions()
options.h = 0.01
options.iterations = 100 / options.h
options.thermostat_strength = 0.1
density_effect_analysis(options)

run_barostat_strengths_test()
