# Standard Library Imports
import random  # For generating random numbers and choices

# Mathematical and Geometric Calculations
import math  # For mathematical operations like trigonometric functions
import numpy as np  # For numerical operations, array manipulations

# Statistical Analysis
from scipy import stats  # For statistical analysis and pattern recognition

# Optimization Algorithms
from scipy.optimize import minimize  # For optimizing pizza selection and cutting

# Additional Utilities
from collections import defaultdict  # For easier handling of data structures
from typing import List, Tuple, Dict  # For type annotations
from tokenize import String
import constants
from utils import pizza_calculations
from shapely.geometry import LineString, Point
import copy


class Player:
    def __init__(self, num_toppings, rng):
        self.num_toppings = num_toppings
        self.rng = rng
        self.pizza_radius = 6
        self.topping_radius = 0.375
        self.pizza_center = [0, 0]
        self.sequence = 0

    def customer_gen(self, num_cust, rng=None):
        def create_inst():
            # Generate preferences using the beta distribution
            alpha, beta = 2, 2  # You can adjust these parameters as needed
            p = np.random.beta(alpha, beta, self.num_toppings)

            # Scale and normalize preferences to sum to 12, and clamp between 0.1 and 11.9
            p = 11.8 * p / np.sum(p) + 0.1
            return p

        preferences_total = []
        rng = rng if rng is not None else self.rng

        for _ in range(num_cust):
            preferences_1 = create_inst()
            preferences_2 = create_inst()

            preferences = [preferences_1, preferences_2]
            equal_prob = rng.random()
            if equal_prob <= 0.0:  # Change this if you want toppings to show up
                preferences = (np.ones((2, self.num_toppings))
                               * 12 / self.num_toppings).tolist()

            preferences_total.append(preferences)

        return preferences_total

    def choose_toppings(self, preferences):
        # 10 pizzas, 24 toppings each, 3 values per topping (x, y, type)
        pizzas = np.zeros((10, 24, 3))

        pizza_radius = 3
        for j in range(constants.number_of_initial_pizzas):  # Iterate over each pizza
            pizza_indiv = np.zeros((24, 3))

            for i in range(24):  # Place 24 toppings on each pizza
                angle_increment = 2 * np.pi / 24
                angle = i * angle_increment

                # Calculate x, y coordinates
                x = pizza_radius * np.cos(angle)
                y = pizza_radius * np.sin(angle)

                # Assign topping type based on the number of toppings
                if self.num_toppings == 2:
                    topping_type = 1 if i < 12 else 2
                elif self.num_toppings == 3:
                    if i < 8:
                        topping_type = 1
                    elif i < 16:
                        topping_type = 2
                    else:
                        topping_type = 3
                else:  # self.num_toppings == 4
                    if i < 6:
                        topping_type = 1
                    elif i < 12:
                        topping_type = 2
                    elif i < 18:
                        topping_type = 3
                    else:
                        topping_type = 4

                pizza_indiv[i] = [x, y, topping_type]

            pizzas[j] = pizza_indiv

        return list(pizzas)

    def choose_and_cut(self, pizzas, remaining_pizza_ids, customer_amounts):
        best_score = -float('inf')
        best_pizza = None
        best_cut = None
        best_angle = None
        pizza_id = remaining_pizza_ids[0]
        current_pizza = pizzas[pizza_id]
        # Start with center and quadrants
        cut_points = [self.pizza_center] + self.get_quadrant_centers()
        self.sequence = 0

        while self.sequence < 6:
            print("Sequence: " + str(self.sequence))
            new_cut_points = []
            for point in cut_points:
                print(point)
                angle, score = self.find_optimal_cut_angle(
                    current_pizza, point[0], point[1], customer_amounts)
                print(str(angle))
                if score > best_score:
                    best_score = score
                    best_cut = point
                    best_angle = angle
            # Generate new points around the current point for next sequence
            new_cut_points += self.generate_new_points_around(best_cut)
            cut_points = new_cut_points
            self.sequence += 1
            print("Best Cut: " + str(best_cut) +
                  " Best Angle: " + str(best_angle))
        return pizza_id, best_cut, best_angle

    def get_quadrant_centers(self):
        radius = self.pizza_radius / 2  # Half the pizza radius to get quadrant centers
        return [
            [radius, radius],  # Top right quadrant
            [-radius, radius],  # Top left quadrant
            [-radius, -radius],  # Bottom left quadrant
            [radius, -radius]  # Bottom right quadrant
        ]

    def find_optimal_cut_angle(self, pizza, x, y, customer_amounts):
        best_angle = None
        best_score = -float('inf')

        # Assuming multiplier is a constant defined elsewhere in your code
        multiplier = 40  # Your multiplier value

        for angle in [i * math.pi / 36 for i in range(36)]:
            cut = [x, y, angle]
            B, C, _, _, _, _ = self.calculate_pizza_score(pizza, cut, customer_amounts, self.num_toppings, multiplier,
                                                          x, y)
            # Assuming we want to maximize the total improvement
            score = np.sum(B) - np.sum(C)

            if score > best_score:
                best_score = score
                best_angle = angle

        return best_angle, best_score

    def calculate_pizza_score(self, pizza, pizza_cut, preferences, num_toppings, multiplier, x, y):
        # Calculate score for one pizza
        B = []
        C = []
        U = []
        obtained_preferences = []
        center_offset = []
        slice_amount_metric = []

        # Calculate the ratios, areas, and preferences
        pizza_calculator = pizza_calculations()
        obtained_pref, slice_areas_toppings = pizza_calculator.ratio_calculator(
            pizza, pizza_cut, num_toppings, multiplier, x, y)

        obtained_pref = np.array(obtained_pref)
        slice_areas = pizza_calculator.slice_area_calculator(
            pizza_cut, multiplier, x, y)

        # Random preference for U calculation
        random_pref, _ = pizza_calculator.ratio_calculator(
            pizza, [x, y, self.rng.random() * 2 * np.pi], num_toppings, multiplier, x, y)
        random_pref = np.array(random_pref)
        required_pref = np.array(preferences)
        uniform_pref = np.ones((2, num_toppings)) * (12 / num_toppings)

        # Calculate B, C, and U
        b = np.round(np.absolute(required_pref - uniform_pref), 3)
        c = np.round(np.absolute(obtained_pref - required_pref), 3)
        u = np.round(np.absolute(random_pref - uniform_pref), 3)
        B.append(b)
        C.append(c)
        U.append(u)
        obtained_preferences.append(tuple(np.round(obtained_pref, 3)))

        # Extra metrics
        x_offset = (pizza_cut[0] - x) / multiplier
        y_offset = (pizza_cut[1] - y) / multiplier
        center_offset.append(np.sqrt(x_offset ** 2 + y_offset ** 2))
        sum_1, sum_2, sum_metric = 0, 0, 0
        for j, area in enumerate(slice_areas):
            if j % 2 == 0:
                sum_2 += area
            else:
                sum_1 += area

        for k in range(num_toppings):
            for l, area_topping in enumerate(slice_areas_toppings):
                if l % 2 == 0:
                    sum_metric += abs((preferences[1][k] *
                                      slice_areas[l] / sum_2) - area_topping[k])
                else:
                    sum_metric += abs((preferences[0][k] *
                                      slice_areas[l] / sum_1) - area_topping[k])
        slice_amount_metric.append(sum_metric)

        return B, C, U, obtained_preferences, center_offset, slice_amount_metric

    def determine_slice(self, topping, cut_point, cut_angle):
        # Convert topping position to polar coordinates relative to the cut point
        dx = topping[0] - cut_point[0]
        dy = topping[1] - cut_point[1]
        angle = math.atan2(dy, dx)

        # Normalize angle to be between 0 and 2*pi
        angle = angle if angle >= 0 else (2 * math.pi + angle)

        # Determine the starting angle of each slice
        slice_angles = [(cut_angle + i * math.pi / 4) %
                        (2 * math.pi) for i in range(8)]

        # Sort the slice angles and find which range the topping's angle falls into
        slice_angles.sort()
        for i in range(8):
            start_angle = slice_angles[i]
            end_angle = slice_angles[(i + 1) % 8]
            if start_angle < end_angle:
                if start_angle <= angle < end_angle:
                    return i
            else:  # Case where the slice crosses the 0 angle
                if start_angle <= angle or angle < end_angle:
                    return i

        # If for some reason it doesn't fall into any slice, return an error value
        return -1

    def generate_new_points_around(self, point):
        new_points = []
        # Adjust the distance from the original point
        delta = self.pizza_radius / (2 * (self.sequence + 1))

        for dx in [-delta, 0, delta]:
            for dy in [-delta, 0, delta]:
                if dx != 0 or dy != 0:  # Exclude the original point
                    new_points.append([point[0] + dx, point[1] + dy])

        return new_points
