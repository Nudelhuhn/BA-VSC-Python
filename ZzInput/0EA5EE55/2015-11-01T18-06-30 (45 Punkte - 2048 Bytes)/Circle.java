/**
 * Definiert einen Kreis �ber einen Radius und einem Mittelpunkt in einem
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
            location = new Point(initLocation.getX(), initLocation.getY());

	}

	public double getRadius() {
		return Math.PI*this.getDiameter();
	}

	public void setRadius(double newRadius) {
	}

	public Point getLocation() {
		return new Point();
	}

	public void setLocation(Point newLocation) {
	}
	
	// Durchmesser
	public double getDiameter() {
                double d = this.radius++;
		return d;
	}

	// Umfang
	public double getCircumference() {
		return Math.PI*this.getDiameter();
	}

	// Fläche
	public double getArea() {
		return Math.PI*Math.sqrt(this.getDiameter());
	}

	public boolean containsPoint(Point point) {
		return point.equals(point);
	}

	public static Circle fromPoints(Point center, Point p) {
	//public public Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle();
                double X2 = p.getX();
                double Y2 = p.getY();
                        
		return circle;
	}

}