//import java.awt.Point;

/**
 * Definiert einen Kreis über einen Radius und einem Mittelpunkt in einem
 * zweidimensionalen Koordinatensystem
 */
public class Circle {

	// Mittelpunkt des Kreises
	private Point location;
	private double radius;

	public Circle() {
	}

	public Circle(Point initLocation, double initRadius) {
		this.location = initLocation;
		this.radius = initRadius;
	}

	public double getRadius() {
		return this.radius;
	}

	public void setRadius(double newRadius) {
		this.radius = newRadius;
	}

	public Point getLocation() {
		return this.location;
	}

	public void setLocation(Point newLocation) {
		this.location = newLocation;
	}
	
	// Durchmesser
	public double getDiameter() {
		return this.radius * 2;
	}

	// Umfang
	public double getCircumference() {
		return 2 * Math.PI * this.radius;
	}

	// Fläche
	public double getArea() {
		return Math.PI * this.radius * this.radius;
	}

	public boolean containsPoint(Point point) {
		return Math.sqrt(
				Math.pow(this.location.getX()-point.getX(), 2) + Math.pow(this.location.getY()-point.getY(), 2)
				) <= this.radius;
	}

	public static Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle();

		// An dieser Stelle das circle-Objekt bearbeiten
		circle.setLocation(center);
		circle.setRadius(center.getDistance(p));

		return circle;
	}

}