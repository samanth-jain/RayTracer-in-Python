import numpy as np
import matplotlib.pyplot as plt

# Define constants
WIDTH = 400
HEIGHT = 300
FOV = np.pi / 2

# Define colors
WHITE = (1, 1, 0)
BLACK = (0, 0, 0)

# Define sphere properties
sphere_pos = np.array([0, 0, 3])
sphere_radius = 1

# Function to trace rays and determine color
def trace_ray(origin, direction):
    t_sphere = intersect_sphere(origin, direction)
    if t_sphere:
        hit_point = origin + direction * t_sphere
        normal = (hit_point - sphere_pos) / sphere_radius
        brightness = max(0, np.dot(normal, -direction))
        return (brightness, brightness, brightness)  # Return as RGB tuple
    return None

# Function to find intersection with sphere
def intersect_sphere(origin, direction):
    oc = origin - sphere_pos
    a = np.dot(direction, direction)
    b = 2.0 * np.dot(oc, direction)
    c = np.dot(oc, oc) - sphere_radius**2
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return None
    else:
        return (-b - np.sqrt(discriminant)) / (2*a)

# Function to render scene
def render():
    image = np.zeros((HEIGHT, WIDTH, 3))
    for y in range(HEIGHT):
        for x in range(WIDTH):
            ray_dir_x = (2 * (x + 0.5) / WIDTH - 1) * np.tan(FOV / 2) * WIDTH / HEIGHT
            ray_dir_y = -(2 * (y + 0.5) / HEIGHT - 1) * np.tan(FOV / 2)
            ray_dir = np.array([ray_dir_x, ray_dir_y, -1])
            ray_dir /= np.linalg.norm(ray_dir)
            color = trace_ray(np.array([0, 0, 0]), ray_dir)
            if color is not None:
                image[y, x] = np.clip(np.array(color) * WHITE, 0, 1)  # Modify color multiplication
    return image

# Render the scene
image = render()

# Display the rendered image
plt.imshow(image)
plt.show()
