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
		location = initLocation;
		radius = initRadius;
	}

	public double getRadius() {
		return radius;
	}

	public void setRadius(double newRadius) {
		radius = newRadius;
	}

	public Point getLocation() {
		return location;
	}

	public void setLocation(Point newLocation) {
		location = newLocation;
	}
	
	// Durchmesser
	public double getDiameter() {
		return radius*2;
	}

	// Umfang
	public double getCircumference() {
		double u = 2*Math.PI*radius ; // Umfang
		return u;
	}

	// Fläche
	public double getArea() {
		double a= Math.PI*(Math.pow(radius, 2)); //Flächeninhalt
		return a;
	}

	public boolean containsPoint(Point point) {
		if((location.getX()+radius>=point.getX())
				&& (location.getX()-radius<=point.getX())
				&& (location.getX()-radius<=point.getX()) 
				&& (location.getY()+radius>=point.getY()) 
				&& (location.getY()-radius<=point.getY()))
		{
			return true;
		}
		return false;		
	}

	public static Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle ();

		// An dieser Stelle das circle-Objekt bearbeiten

		return circle;
	}

}