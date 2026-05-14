import numpy as np
from PIL import Image, ImageDraw, ImageFont

def array_to_png(arr, pixel_size=50, filename='output.png'):
	"""Convert a 1D or 2D array of values 1 and -1 into a PNG image.

	- `arr` can be a Python list, a list of lists (2D), or a `numpy.ndarray`
	  containing only values 1 or -1.
	- `pixel_size` is the size (in pixels) of each square cell in the grid (default: 50).
	- `filename` is the output PNG file path.

	Returns the saved file path.
	"""
	is_numpy = hasattr(arr, 'ndim')
	if is_numpy:
		a = arr
	else:
		a = np.array(arr)

	if a.ndim == 1:
		rows, cols = 1, a.shape[0]
		grid = a.reshape((1, cols))
	elif a.ndim == 2:
		rows, cols = a.shape
		grid = a
	else:
		raise ValueError('Array must be 1D or 2D')

	unique = np.unique(grid)
	if not set(unique.tolist()).issubset({1, -1}):
		raise ValueError('Array values must be only 1 or -1')

	# map 1 -> 255 (white), -1 -> 0 (black)
	mapped = np.where(grid == 1, 255, 0).astype(np.uint8)

	img = Image.fromarray(mapped, mode='L')
	img = img.resize((cols * pixel_size, rows * pixel_size), Image.NEAREST)
	# If a filename is provided, save to disk and return the path.
	# Otherwise, return the PIL Image object for in-memory use.
	if filename:
		img.save(filename, format='PNG')
		return filename
	return img


def animate(arrays, pixel_size=50, filename='animation.gif', fps=5, loop=0, cleanup=True):
	"""Create an animated GIF from a sequence of 1D/2D arrays.

	- `arrays` is an iterable of arrays (lists or numpy arrays) where each
	  element is a 1D or 2D array of values 1 and -1.
	- `pixel_size` sets the size of each square cell (default: 50).
	- `filename` is the output animation path (GIF).
	- `fps` frames per second; used to compute per-frame duration.
	- `loop` number of loops for the GIF (0 means infinite).
	- `cleanup` whether to remove temporary frame files after creation (deprecated, ignored).

	Adds frame counter "n/N" to each frame.
	Returns the animation filename.
	"""

	print(f"Creating animation with {len(arrays)} frames at {fps} fps...")
	
	if fps <= 0:
		raise ValueError('`fps` must be a positive number')
	duration_ms = int(1000 / fps)

	# Convert to list to get total frame count
	arrays_list = list(arrays)
	total_frames = len(arrays_list)

	if not arrays_list:
		raise ValueError('No frames provided')

	frames = []
	# Try to load a larger font; fall back to default
	try:
		font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 20)
	except Exception:
		try:
			font = ImageFont.truetype('arial.ttf', 20)
		except Exception:
			font = ImageFont.load_default()

	for frame_idx, arr in enumerate(arrays_list):
		img = array_to_png(arr, pixel_size=pixel_size, filename=None)
		
		# Convert to RGB for colored text
		img_rgb = img.convert('RGB')
		
		# Add frame counter text in red
		draw = ImageDraw.Draw(img_rgb)
		text = f'{frame_idx + 1}/{total_frames}'
		# Position text in top-left corner with small margin
		draw.text((5, 5), text, fill=(255, 0, 0), font=font)
		                                 
		# ensure palette mode for GIF
		frames.append(img_rgb.convert('P'))


	frames[0].save(
		filename,
		save_all=True,
		append_images=frames[1:],
		duration=duration_ms,
		loop=loop,
	)
	return filename



if __name__ == '__main__':
	demo1 = [1, -1, 1, 1, -1, -1, 1]
	array_to_png(demo1, pixel_size=40, filename='demo_row.png')

	demo2 = [[1, -1, 1, -1], [ -1, 1, -1, 1], [1, 1, -1, -1]]
	array_to_png(demo2, pixel_size=30, filename='demo_grid.png')


