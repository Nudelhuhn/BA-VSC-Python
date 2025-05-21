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
		//funktioniert auch ohne this.
		this.x = initX;
		this.y= initY;
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
		double d = Math.sqrt(Math.pow((p.getX()-this.getX()), 2)+(Math.pow((p.getY()-this.getY()), 2)));
		return d;
	}
}