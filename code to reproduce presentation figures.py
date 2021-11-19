import time
from advection import Advection

# numpy and matplotlib are required

tstart = time.perf_counter()
print('Program started')
directory = '1D_advection'
advection = Advection(dx=0.001, animation_directory=directory, plot_directory=directory, record_to_file=False)

advection.plot(plot_expected=False)
advection.full_expected()
advection.create_animation(savefile='Ideal Results', plot_expected=False, frame_spacing=100)

advection.set_dx(0.01)
advection.forward_time_central_space()
advection.plot()  # This plots last frame - advection.plot must be called before animation
advection.create_animation(frame_spacing=100)

advection.set_dx(0.001)
advection.lax()
advection.plot()
advection.create_animation(frame_spacing=100)

#  To use different methods without changing variables, call advection.reset() to clear data
advection.reset()
advection.lax_wendroff()
advection.plot()
advection.create_animation(frame_spacing=100)

timeElapsed = round(time.perf_counter() - tstart, 4)
print(f'Program closing - took {timeElapsed}s, ')
