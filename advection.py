import os.path
import sys
import time
from datetime import datetime
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as plt_ticker


# code 7 hours - 1 hour on presentation and getting results

class Advection:
    """A class for solving, plotting, and animating the 1d linear advection equation with periodic boundary
conditions, and rectangle pulse, using various numerical methods,with automatic timing and
data recording abilities. important variables, like x step size, t step size, x axis width, pulse width
directories where plots, results and animations are saved, where the run time of methods are saved
and if results of methods are recorded can be changed using the appropriate setters"""

    def __init__(self, dx: float = 0.01, dt: float = 0.0001, t_max: float = 1, c: float = 1.0,
                 pulse_width: float = 0.5, x_axis_width: float = 1.0, timer_filename: str = "advection_timer.txt",
                 animation_directory: str = 'advection_animations', plot_directory: str = 'advection_plots',
                 results_directory: str = 'advection_results', record_to_file: bool = True):
        """Initialises the advection class - setting values and initial array, and printing set values to screen"""
        try:
            self.record_to_file = bool(record_to_file)  # If true data of methods record to txt file
            self.method_title = 'Initial Conditions'
            self.t_max = float(t_max)  # Time in seconds the methods are run until
            self.half_pulse_width = float(
                pulse_width) / 2  # width of rectangle pulse in initial conditions centered on x = 0
            self.x_axis_width = float(x_axis_width)  # width of x axis, the smaller the faster that the pulse will run
            # into the noise it created in its wake
            self.set_dx(float(dx))  # x spacing - time complexity O(t/dt * x/dx)
            self.set_dt(float(dt))  # time spacing  - time complexity O(t/dt * x/dx)
            self.set_c(float(c))  # speed of pulse
            self.yAxisData = []  # array to record values at end of each iteration in time
            self.set_time_filename(timer_filename)  # filename that the run time of each method recorded to
            self.set_results_directory(results_directory)  # directory method results recorded to
            self.set_animation_directory(animation_directory)  # directory animations saved to
            self.set_plot_directory(plot_directory)  # directory plots saved to
        except Exception as e:
            print('Error creating advection object, check types of arguments')
            print(e)
            exit()
        print(
            f'Advection class created with - maxt {self.t_max}s, c - {self.c} dt = {self.dt}, and dx = {self.dx} x axis width - {self.x_axis_width}'
            f'with initial square pulse and periodic boundary conditions of pulsewidth {self.half_pulse_width * 2}')
        print(f'Class loaded at {self.__get_datetime()} recording results to {self.timer_filename}')

    def __timer(method):
        """Timer decorator, prints name of decorated method and when it was called,
        prints when it is finished, and how long it took
        records name of method, how long it took and some variable values to timer_filename.txt"""

        def timed(self, *args, **kwargs):
            time_start = time.perf_counter()
            methodname = self.__clean_method_name(method.__name__)
            self.__announce_started(methodname)
            timedResult = method(self, *args, **kwargs)
            time_elapsed = round(time.perf_counter() - time_start, 4)
            if not os.path.exists(self.timer_filename):
                with open(self.timer_filename, 'w') as f:
                    f.write("%-28s %-10s %-7s %-7s %-6s %-7s %-7s  %-20s" % (
                        'Name', 'RunTime(s)', 'MaxT(s)', 'dt(s)', 'Points', 'dx(m)', 'C(m/s)', 'TimeRun'))
                    print(f'{self.timer_filename} created')
            with open(self.timer_filename, 'a') as f:
                f.write(
                    f"\n{methodname:28} {time_elapsed:-10} {self.t_max:-7} {self.dt:-7} {self.points:-6} {self.dx:-8} {self.c:-7}  {self.__get_datetime():25}")
            current = self.__get_time()
            print(f'{methodname} finished at {current}, taking {time_elapsed}s')
            return timedResult

        return timed

    @staticmethod
    def __error_message(error_type: str, name1: str, name2: str) -> str:
        """Prints error message for given type"""
        error_type = error_type.lower()
        if error_type == 'larger':
            return f'{name1} can not be larger than {name2}'

    @staticmethod
    def __clean_method_name(name: str) -> str:
        """Cleans method names for user output, removing underscores and capitalising"""
        name = name.replace("_", " ")
        name = name.title()
        return name

    @staticmethod
    def __get_time() -> str:  # group with dictionary
        """Returns time in %H:%M:%S format"""
        return datetime.now().strftime("%H:%M:%S")

    @staticmethod
    def __get_file_time() -> str:
        """Returns time in %H_%M_%S_%d_%m_%Y format for saving files"""
        return datetime.now().strftime("%H_%M_%S_%d_%m_%Y")

    @staticmethod
    def __get_datetime() -> str:
        """Returns time in %H:%M:%S %d/%m/%Y format for printing to screen"""
        return datetime.now().strftime("%H:%M:%S %d/%m/%Y")

    def __announce_started(self, name: str):
        """Prints that a method has started, and the time it did so"""
        now = self.__get_time()
        print(f'{name} started at {now}')

    def __store_grid_data(self) -> None:
        """Stores the grid data to yaxisdata variable for later plotting,
        if record to file is true, it will also record the data to a txt file"""
        self.yAxisData.append(self.grid.copy())
        if self.record_to_file:
            self.__record_to_file()

    def __open_new_record_file(self) -> None:
        """Opens new file to record results of method to, saving it to variable if it does not exist, it is created"""
        self.record_filename = self.results_directory + self.method_title + "_" + self.record_time + '.txt'
        self.record_filename = self.record_filename.replace(" ", "_")
        if not os.path.exists(self.record_filename):
            with open(self.record_filename, 'w') as f:
                pass
        self.record_file_handle = open(self.record_filename, 'a')
        print(f'Recording {self.method_title} results to {self.method_title.replace(" ", "_") + "_" + self.record_time}.txt')

    def __close_record_file(self) -> None:
        """Closes record file handle"""
        self.record_file_handle.close()

    def __record_to_file(self) -> None:
        """Records current grid array to record file"""
        self.grid.tofile(self.record_file_handle, sep=',', format='%s')
        self.record_file_handle.write("\n")

    def __update_method_title(self, name) -> None:
        """To be called at the start of a method - stores the name of the method to allow
        graphs later created to be automatically named appropriately"""
        self.method_title = self.__clean_method_name(name)

    def __change_directory(self, type, directory: str = None) -> None:
        """Checks new directory name is appropriate - creates it if it does not exist,
        and stores the directory path in the appropriate variable"""
        if directory is None:
            print(f'{type} directory name not changed - please provide a valid string')
            return
        if not os.path.isdir(directory):
            os.mkdir(directory)
        if directory[0] != '\\':
            directory = '\\' + directory
        if directory[-1] != '\\':
            directory = directory + '\\'
        if ':' not in directory:
            directory = os.getcwd() + directory
        print(f'{type.capitalize()} directory - {directory}')
        if type == 'animation':
            self.animation_directory = directory
            return
        elif type == 'plot':
            self.plot_directory = directory
            return
        elif type == 'results':
            self.results_directory = directory
            return
        print('Invalid type provided, no directory changed')

    def __get_plus_one(self, i: int) -> int:
        """Stores appropriate index for i + 1 to allow loop around and prevent index errors - to be used
        for each value of i, before equation"""
        if i == self.points - 1:
            self.plus_one = 0
        else:
            self.plus_one = i + 1
        return self.plus_one

    def __update_previous(self) -> None:
        """Stores values of previous grid - to be used at end of method"""
        self.previous = self.grid.copy()

    def __create_initial_grid(self) -> None:
        """Creates the initial grid with initial conditions"""
        self.initial_grid = np.zeros(self.points)
        upperbound = (self.x_axis_width / (2 * self.dx)) + (self.half_pulse_width / self.dx)
        lowerbound = (self.x_axis_width / (2 * self.dx)) - (self.half_pulse_width / self.dx)
        for index, value in enumerate(self.initial_grid):
            if upperbound > index > lowerbound:
                self.initial_grid[index] = 1
            if index * self.dx > upperbound:
                break

    def __initiate_grid(self) -> None:
        """Initiates the initial grid and calculate number of points"""
        self.points = int(self.x_axis_width / self.dx) + 1
        self.__create_initial_grid()
        self.reset()

    def __set_x_axis_data(self) -> None:
        """Creates array for x values of appropriate length - to allow proper x values to be written on x axis
        of plots"""
        self.xAxisData = np.arange(-self.x_axis_width / 2, self.x_axis_width / 2 + self.dx, self.dx)

    def __start_method(self, name) -> None:
        """Calculates courant number, stores title of method - to be used in plot titles, and opens new file to record
        data to"""
        self.record_time = self.__get_file_time()
        self.__set_half_courant_number()
        self.__set_half_courant_number_squared()
        self.__update_method_title(name)
        if self.record_to_file:
            self.__open_new_record_file()

    def __close_method(self) -> None:
        """Stores final data for plotting and closes file handle"""
        self.__store_grid_data()
        if self.record_to_file:
            self.__close_record_file()

    def __set_half_courant_number(self) -> None:
        """Calculates half the courant number"""
        self.half_courant = (self.c * self.dt) / (2 * self.dx)

    def __set_half_courant_number_squared(self):
        """Calculates half of (the courant number squared) C^2/2"""
        self.half_courant_squared = self.half_courant * self.half_courant * 2

    def set_time_filename(self, filename: str = None) -> None:
        """Sets file name where run times of methods are recorded"""
        if filename is None:
            print(
                f'Time results file name not changed - please provide a valid string - recording results to {self.timer_filename}')
        else:
            self.timer_filename = filename
            print(f'Now recording results to {self.timer_filename}')

    def set_plot_directory(self, directory: str = None) -> None:
        """Change directory where plots are saved"""
        self.__change_directory('plot', directory)

    def set_animation_directory(self, directory: str = None) -> None:
        """Changed directory where animations are saved"""
        self.__change_directory('animation', directory)

    def set_results_directory(self, directory: str = None) -> None:
        """Changes directory where results are saved"""
        self.__change_directory('results', directory)

    def set_record_to_file(self, record: bool = True) -> None:
        """Changes whether results of numerical methods are recorded to file"""
        self.record_to_file = bool(record)

    def set_maxt(self, maxt: float) -> None:
        """Changes maxt - the time the method will run until
        - doubling this value doubles time complexity and space complexity"""
        if maxt < self.dt:
            raise Exception(self.__error_message('larger', 'dt', 'maxt'))
        self.t_max = float(maxt)
        self.__initiate_grid()

    def set_c(self, c: float) -> None:
        """Changes d, and resets grid"""
        self.c = c
        self.reset()

    def set_x_axis_width(self, x_axis_width: int) -> None:
        """Changes initial width of x axis, and resets grid to include new initial conditions
        - the width is the total width centered on 0 e.g. 2 goes from -1 to 1
        - doubling this value doubles time complexity and space complexity"""
        if self.half_pulse_width > x_axis_width:
            raise Exception(self.__error_message('larger', 'pulsewidth', 'x_axis_width'))
        self.x_axis_width = x_axis_width
        self.__initiate_grid()

    def set_pulsewidth(self, pulsewidth: float) -> None:
        """Changes pulse width, and resets grid to include new initial conditions"""
        if pulsewidth > self.x_axis_width:
            raise Exception(self.__error_message('larger', 'pulsewidth', 'x_axis_width'))
        self.half_pulse_width = pulsewidth / 2
        self.__initiate_grid()

    def set_dx(self, dx: float) -> None:
        """Changes dx, and resets grid to include new initial conditions
        - halving this value doubles the time complexity and space complexity"""
        if dx > self.x_axis_width:
            raise Exception(self.__error_message('larger', 'dx', 'x_axis_width'))
        self.dx = dx
        self.__initiate_grid()

    def set_dt(self, dt: float) -> None:
        """Changes dt, and resets grid
        - halving this value doubles the time complexity and space complexity"""
        if dt > self.t_max:
            raise Exception(self.__error_message('larger', 'dt', 'maxt'))
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        """Resets grid to initial values and clears data stored in memory
        - should be called by user after a method has finished or when important variables are changed
        Does not restore variables to default"""
        try:
            self.grid = self.initial_grid.copy()
        except:
            self.__create_initial_grid()
            self.grid = self.initial_grid.copy()
        self.previous = self.grid.copy()
        self.yAxisData = []
        self.yAxisData.append(self.grid.copy())
        self.method_title = 'Initial Conditions'

    def expected_result_at_t(self, current_time=None) -> None:
        """Calculates the expected result of the 1d linear advection equation for a given time"""
        if current_time is None:
            current_time = self.t_max
        for i in range(self.points):
            index = int((i - (self.c / self.dx) * current_time)) % self.points
            self.grid[i] = self.initial_grid.copy()[index]

    @__timer
    def full_expected(self) -> None:
        """Calculates the expected result of the 1d linear advection over time period"""
        self.__start_method(sys._getframe().f_code.co_name)
        t = 0
        while self.t_max >= t:
            self.__store_grid_data()
            t += self.dt
            self.expected_result_at_t(t)
        self.__close_method()

    def __base_method(self, name) -> None:
        """Base method for all methods - implements loops, file openings and recordings"""
        methods = {'forward_time_central_space': self.__forward_time_central_space_equation,
                   'lax': self.__lax_equation,
                   'lax_wendroff': self.__lax_wendroff_equation}
        self.__start_method(name)  # opens file calculates courant numbers
        t = 0
        equation = methods[name]
        while self.t_max > t:
            self.__store_grid_data()  # stores initial data and intermediary data
            for i in range(self.points):
                self.__get_plus_one(i)
                equation(i)
            self.__update_previous()
            t += self.dt
        self.__close_method()  # stores final data and closes file etc.

    @__timer
    def forward_time_central_space(self) -> None:
        """Solves 1D linear advection equation using forward time central space method"""
        self.__base_method(sys._getframe().f_code.co_name)

    def __forward_time_central_space_equation(self, i: int) -> None:
        """Equation for forward time central space"""
        self.grid[i] = self.previous[i] + self.half_courant * (
                self.previous[i - 1] - self.previous[self.plus_one])

    @__timer
    def lax(self) -> None:
        """Solves 1D linear advection equation using lax method"""
        self.__base_method(sys._getframe().f_code.co_name)

    def __lax_equation(self, i: int) -> None:
        """Equation for lax scheme"""
        self.grid[i] = ((self.previous[self.plus_one] + self.previous[i - 1]) / 2) - (
                self.half_courant * (self.previous[self.plus_one] - self.previous[i - 1]))

    @__timer
    def lax_wendroff(self) -> None:
        """Solves 1D linear advection equation using lax-wendroff method"""
        self.__base_method(sys._getframe().f_code.co_name)

    def __lax_wendroff_equation(self, i: int) -> None:
        """equation for lax wendroff scheme"""
        self.grid[i] = self.previous[i] - self.half_courant * (
                self.previous[self.plus_one] - self.previous[i - 1]) + (
                               self.half_courant_squared *
                               (self.previous[self.plus_one] - 2 * self.previous[i] + self.previous[i - 1]))

    def plot(self, title: str = False, plot_expected: bool = True) -> None:
        """Plots data currently stored in the grid value - this will be the value of u(x, t) at maxt
        plots are saved to plot directory, title is the plot title and the name of the file
        it is saved to - !!Must be called before any animations are created!! """
        if title is False:
            title = self.method_title
        fig, ax = plt.subplots()
        self.__set_x_axis_data()
        time_created = self.__get_file_time()
        ax.clear()
        ax.set_xlabel('X')
        if self.method_title != 'Initial Conditions':
            ax.set_ylabel(f'u(x,t = {self.t_max})')
        else:
            ax.set_ylabel('u(x,t = 0)')
        ax.set_ylim(0, 1.4)
        ax.set_xlim(-self.x_axis_width / 2, self.x_axis_width / 2)
        ax.plot(self.xAxisData, self.grid, label='Result')  # After method, data in grid is values for t_max
        if plot_expected:
            self.expected_result_at_t()  # Calculates expected position of t at t_max
            ax.plot(self.xAxisData, self.grid, label='Expected')
        ax.legend(loc='upper right', shadow=False, fontsize='medium')
        ticker = plt_ticker.MultipleLocator(base=self.x_axis_width / 10)
        ax.xaxis.set_major_locator(ticker)
        ax.annotate(f'1D Linear Advection Equation - c-{self.c}; dx-{self.dx}; dt-{self.dt}',
                    xy=(1, -0.15), xycoords='axes fraction', ha='right', va="center", fontsize=9)
        fig.tight_layout()
        ax.set_title(title)
        save_title = title.replace(' ', '_')
        save_title += '_'
        plt.savefig(f'{self.plot_directory}{save_title}{time_created}.png')
        plt.show()
        print(f'{title} plot saved as {save_title}{time_created}.png')

    @__timer
    def create_animation(self, savefile: str = False, frame_spacing: int = 50, plot_expected: bool = True) -> None:
        """Creates an animation of the previous method run - showing the values of u(x,t) over time.
        frame spacing controls how many frames in animations a frame spacing of 2 would print every other frame
        for the time period it has been calculated for, save file can change the name of the file the
        animation is saved to. animations are saved to animation directory - !!Clears data stored in grid variable!!
        """
        extension = '.gif'
        frame_count = int((self.t_max / frame_spacing) / self.dt)
        fps = int((1 / frame_spacing) / self.dt)
        writer = animation.PillowWriter(fps=fps)
        time_created = self.__get_file_time()
        datetime_created = self.__get_datetime()

        if savefile is False:
            title = f'Solution to 1D Linear Advection Equation Using {self.method_title} Method'
            savefile = self.animation_directory + self.method_title.replace(" ", "_") + "_" + time_created + extension
        else:
            title = savefile
            savefile = self.animation_directory + savefile + extension

        print(f'Creating {savefile}')
        self.__set_x_axis_data()
        print(f'Frame spacing - {frame_spacing} intervals or {round(frame_spacing * self.dt, 3)}s')
        print(f'Expecting {frame_count} frames, {fps} frames per second of simulation time, over  {self.t_max}s')
        ticker = plt_ticker.MultipleLocator(
            base=self.x_axis_width / 10)  # Variable used later to place x ticks every 10%

        def animate(i: int) -> None:
            """Function called to plot frame of animation where i is number of the frame"""
            ax.clear()
            i = i * frame_spacing
            plot_time = "%.2f" % round(i * self.dt, 2)
            ax.set_xlabel('X')
            ax.set_ylabel(f'u(x,t = {plot_time})')
            ax.set_ylim(0, 1.4)
            ax.set_xlim(-self.x_axis_width / 2, self.x_axis_width / 2)
            ax.set_title(title)
            ax.title.set_size(9)
            ax.xaxis.set_major_locator(ticker)  # Sets x axis ticks
            ax.plot(self.xAxisData, self.yAxisData[i], label='Result')
            if plot_expected:
                self.expected_result_at_t(i * self.dt)
                ax.plot(self.xAxisData, self.grid, label='Expected')
            ax.legend(loc='upper right', shadow=False, fontsize='medium')
            ax.annotate(
                f'{datetime_created} c-{self.c} dx-{self.dx} dt-{self.dt} FinalT-{self.t_max}s Frame Spacing-{frame_spacing} Frames-{frame_count}',
                xy=(1, -0.15), xycoords='axes fraction', ha='right', va="center", fontsize=9)
            fig.tight_layout()

        fig, ax = plt.subplots()
        ani = animation.FuncAnimation(fig, func=animate, frames=frame_count)
        ani.save(savefile, writer=writer)
        print(f'{savefile} saved')
