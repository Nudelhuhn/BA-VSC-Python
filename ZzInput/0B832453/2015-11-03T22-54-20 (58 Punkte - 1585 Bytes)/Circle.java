
public class Circle {

	
	private Point location;
	private double radius;

	public Circle() {
	}

	public Circle(Point initLocation, double initRadius) {
		location=initLocation;
		radius=initRadius;

	}

	public double getRadius() {
		return -1.0;
	}

	public void setRadius(double newRadius) {
		radius=newRadius;
	}

	public Point getLocation() {
		return new Point();
	}

	public void setLocation(Point newLocation) {
		location=newLocation;
	}
	
	
	public double getDiameter() {
		return 2*getRadius();
	}

	
	public double getCircumference() {
		return 2* Math.PI *radius;
	}

	
	public double getArea() {
		return Math.PI*Math.pow(radius, 2);
		
	}

	public boolean containsPoint(Point point) {
		if(point.getDistance(location)>=radius);
		return true;
	
	} 
	

	public static Circle fromPoints(Point center, Point p) {
		Circle circle = new Circle();
circle.setLocation(center);
double radius=center.getDistance(p);
circle.setRadius(radius);

		

		return circle;
	}

}

