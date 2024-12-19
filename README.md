
# Multisample Anti-Aliasing (MSAA) with Custom Graphics Pipeline

This project demonstrates the implementation of **Multisample Anti-Aliasing (MSAA)** using a custom graphics pipeline. The aim is to improve visual quality in 3D rendering by minimizing aliasing artifacts like jagged edges.

## Description

The project involves a custom-built graphics pipeline to render 3D models with MSAA. It utilizes core Python libraries and user-defined modules such as `Camera`, `Projection`, and `GraphicPipeline`. The implementation focuses on improving rendering quality while maintaining flexibility and performance.

## Features

- **Custom Graphics Pipeline**: Implements a unique pipeline for 3D rendering.
- **Anti-Aliasing Support**: Enhances rendering quality using MSAA.
- **Adjustable Parameters**: Camera and projection parameters can be easily customized.
- **Lighting Effects**: Includes basic lighting using a point light source.

## Installation

### Prerequisites
- Python 3.8+
- Required libraries: `numpy`, `matplotlib`

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/SanaullahKayani/MSAA.git
   cd multisample-anti-aliasing

2. Install the required libraries:
   ```bash
   pip install numpy matplotlib
   ```

3. Ensure the file `line.ply` (or another `.ply` file) is available in the project directory.

4. Run the index file:
   ```bash
   python index.py
   ```

## File Structure

- **index.py**: Main script to set up the camera, projection, and pipeline, and render the 3D model.
- **graphicPipeline.py**: Contains the implementation of the graphics pipeline.
- **camera.py**: Manages the camera's position and orientation.
- **projection.py**: Handles the projection matrix for rendering.
- **readply.py**: Reads vertex and triangle data from a `.ply` file.

## Usage

1. Modify the camera's position, orientation, or projection parameters in `index.py` as needed:
   ```python
   cameraPosition = np.array([1.1, 1.1, 1.1])
   lookAt = np.array([-0.577, -0.577, -0.577])
   up = np.array([0.33333333, 0.33333333, -0.66666667])
   right = np.array([-0.57735027, 0.57735027, 0.])
   ```

2. Adjust the field of view (FOV), aspect ratio, near, and far plane settings:
   ```python
   nearPlane = 0.1
   farPlane = 10.0
   fov = 1.91986
   aspectRatio = width / height
   ```

3. Execute the script and visualize the rendered output using Matplotlib.

## Visuals

### Example Output
Rendered output is displayed using Matplotlib:
```python
import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()
```

## Project Highlights

- Custom-built from the ground up for better understanding of 3D graphics.
- Lightweight and minimal dependencies.
- Flexible enough for various 3D rendering tasks.

## Roadmap

- Add more advanced lighting models (e.g., Phong or Blinn-Phong shading).
- Implement texture mapping for 3D models.
- Integrate real-time interactivity to modify camera and light settings.

## Acknowledgment

This project was developed as part of an academic exercise in 3D graphics rendering. Special thanks to the collaborators and instructors for their support and feedback.

---

Happy rendering!
```
