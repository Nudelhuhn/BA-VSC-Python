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
		this.radius = newRadius;
	}

	public Point getLocation() {
		return location;
	}

	public void setLocation(Point newLocation) {
		this.location = newLocation;
	}
	
	// Durchmesser
	public double getDiameter() {
		return radius*2;
	}

	// Umfang
	public double getCircumference() {
		double u = Math.PI*getDiameter();
		//double u= 2*Math.PI*radius ;
		return u;
	}

	// Fläche
	public double getArea() {
		double a= Math.PI*(Math.pow(radius, 2)); //Flächeninhalt
		return a;
	}

	public boolean containsPoint(Point point) {
		
		//if((location.getX() - radius)< point.getX() && point.getX() < (location.getX() + radius) && (location.getY() - radius)< point.getY() && point.getY() < (location.getY() + radius))
	//	if ((Math.pow((point.getX()-location.getX()),2))+(Math.pow((point.getY()-location.getY()),2))<=Math.pow(radius, 2))
		
		{
	//		return true;
		}
		//return false;	
		return point.getDistance(location) >= getRadius();
	}

	public static Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle ();		

		// An dieser Stelle das circle-Objekt bearbeiten
		circle.setLocation(center);
		circle.setRadius(center.getDistance(p));

		return circle;
	}

}