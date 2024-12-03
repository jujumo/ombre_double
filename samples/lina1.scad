
difference()
{
cube([160, 160, 1.2], center=true);
scale(6) translate([-12, -12, -1]) 
linear_extrude(4)
    import("lina1_stripes.svg");

};