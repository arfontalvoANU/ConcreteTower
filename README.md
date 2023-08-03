# Tower Design Python Script (ConcreteTower)

This Python script is used for tower design calculations based on ACI 307-08 guidelines. The script includes various methods to determine different design parameters for the tower, such as maximum load, salt density and enthalpy, sodium density and enthalpy, receiver weight, and bending resistance of the tower's concrete shell cross-section.

## Requirements

- Python 3.x
- NumPy library
- SciPy library

## Getting Started

1. Clone the repository to your local machine:

git clone https://github.com/arfontalvoANU/ConcreteTower.git


2. Install the required libraries using pip:

pip install numpy scipy


3. Run the `tower_design.py` script:

python TowerDesign.py


## Usage

The `TowerDesign.py` script provides various methods for tower design calculations. You can call these methods with appropriate parameters to obtain the desired design parameters. The methods available in the script are as follows:

1. `max_load(D, W, E, load_type)`: Determines the maximum factored load for design purposes based on ACI 307-08 guidelines. Parameters `D`, `W`, and `E` are arrays representing dead loads, wind loads, and earthquake loads, respectively. `load_type` is an integer value indicating the type of load (1 for wind only, 2 for seismic only, and 3 for wind and seismic).

2. `salt(T)`: Computes the salt density and enthalpy for receiver weight based on specified temperature `T`.

3. `sodium(T)`: Computes the sodium density and enthalpy for receiver weight based on specified temperature `T`.

4. `scaling_w_receiver()`: Calculates the weight of the receiver based on its capacity, geometry, heat transfer fluid (HTF), and operation temperatures.

5. `neutral_axis_angle(load, rhot, rm_tow, t_tow, alpha)`: Computes the neutral axis angle for the tower's concrete shell cross-section subjected to bending moment. Parameters `load`, `rhot`, `rm_tow`, `t_tow`, and `alpha` are required for the calculation.

6. `bending_resistance(rhot, rm_tow, t_tow, alpha)`: Computes the bending resistance of the tower's concrete shell cross-section subjected to bending moment. Parameters `rhot`, `rm_tow`, `t_tow`, and `alpha` are required for the calculation.

## License

This project is licensed under the LGPL-2.1 License - see the [LICENSE.md](LICENSE.md) file for details.

## Contributing

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make changes and commit them with descriptive commit messages.
4. Push your changes to your fork.
5. Submit a pull request.

## Authors

- Your Name (@arfontalvoANU) - [GitHub Profile](https://github.com/arfontalvoANU)

## Acknowledgments

- The ACI 307-08 guidelines for tower design.
- The ASCE 7-16 guidelines for tower design.

