/**
 * Beschreibt einen Punkt im zweidimensionalen Koordinatensystem mit x- und
 * y-Wert
 */

//Aggregation - ist inhalten in, mehrere Aufteilungen

//Komposition - ein Betsandteil von, Objekt besteht aus

public class Point {

	// Koordinaten
	private double x;
	private double y;

	public Point() {
	}

	public Point(double initX, double initY) {
		x = initX;
		y = initY;

	}

	public double getX() {
		return x;
	}

	public void setX(double newX) {
		x = newX;
	}

	public double getY() {
		return y;
	}

	public void setY(double newY) {
		y = newY;
	}

	public double getDistance(Point p) {
		double distance = Math.sqrt(Math.pow((x - p.getX()),2) + Math.pow(y - p.getY(), 2));
		return distance;
	}
}
