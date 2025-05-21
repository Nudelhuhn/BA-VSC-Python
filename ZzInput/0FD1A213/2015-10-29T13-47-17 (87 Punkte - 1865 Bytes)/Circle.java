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
		
		location= initLocation;
		radius= initRadius;

	}

	public double getRadius() {
		return radius;
	}

	public void setRadius(double newRadius) {
		radius= newRadius;
	}

	public Point getLocation() {
		return location;
	}

	public void setLocation(Point newLocation) {
		location= newLocation;
	}
	
	// Durchmesser
	public double getDiameter() {
		
		double diameter= radius*2;
		
		return diameter;
	}

	// Umfang
	public double getCircumference() {
		
		double circumference= Math.PI*(radius*2);
		
		return circumference;
	}

	// Fläche
	public double getArea() {
		
		double area= Math.PI*(Math.pow(radius, 2));
		
		return area;
	}

	public boolean containsPoint(Point point) {
		return false;
	}

	public static Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle();

		// An dieser Stelle das circle-Objekt bearbeiten

		
		
		return circle;
	}

}
