import csv
from collections import namedtuple

from flask import Flask, request
from flask_cors import CORS
from geopy import distance

app = Flask(__name__, static_folder="./build", static_url_path="/")
CORS(app)

Airport = namedtuple("Airport", "name ICAO latitude longitude")
airports = {}

# Load airports from file
with open("uk_airport_coords.csv") as file:
    csv_reader = csv.reader(file, delimiter=",")
    lines = 0
    for line in csv_reader:
        if lines != 0:
            airports[line[0]] = Airport(line[0], line[1], line[2], line[3])
        lines += 1


def find_closest_airport(latitude: float, longitude: float) -> dict:
    """Accepts a location and finds the nearest airport from the provided list

    Args:
        latitude (float): The input latitude
        longitude (float): The input longitude

    Returns:
        dict: Dictionary with the nearest airport, distance from given location to the airport
    """
    minimum_distance = -1
    closest_airport_key = ""

    for name, airport in airports.items():
        delta = distance.distance(
            (latitude, longitude), (airport.latitude, airport.longitude)
        )
        if minimum_distance > delta or minimum_distance == -1:
            minimum_distance = delta
            closest_airport_key = name

    return {
        "airport": airports.get(
            closest_airport_key
        )._asdict(),  # As dict to better support Flask jsonify
        "distance_km": minimum_distance.kilometers,
    }


def parse_args(args: dict) -> None:
    """Parse arguments from endpoint and raise appropriate exceptions if not valid

    Args:
        args (dict): arguments to be parsed

    Raises:
        Exception: _description_
    """
    if "latitude" not in args or "longitude" not in args:
        raise Exception("Not all required arguments provided")
    try:
        return float(args["latitude"]), float(args["longitude"])
    except ValueError:
        raise Exception("Could not parse one of the values")


@app.route("/nearest_airport", methods=["GET"])
def nearest_airport():
    try:
        coords = parse_args(request.args)
    except Exception as e:
        return str(e), 400
    return find_closest_airport(coords[0], coords[1])


@app.route("/")
def index():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(port=8000, debug=True)
