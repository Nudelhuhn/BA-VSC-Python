/**
 * Beschreibt einen Punkt im zweidimensionalen Koordinatensystem mit x- und
 * y-Wert
 */
public class Point {

	private static int initX;
	// Koordinaten
	private double x;
	private double y;
	private int initY;

	public Point() {
	}

	public Point(double initX, double initY) {
		x = initX;
		y= initY;
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
		double d = Math.sqrt(Math.pow((p.getX()-p.getX()), 2)+(Math.pow((p.getY()-p.getY()), 2)));
		return d;
	}
}