#!/usr/bin/perl -w
use lib qw(../../L80G);
use Math::Trig;
use Universe::Distance;

my $LK_TO_LC = log(1/1.07)/log(10);
my $pr_lim = 17.77;

open REGIONS, "<auto_regions.txt";
while (<REGIONS>) {
    my @a = split;
    last if (/END/);
    push @regions, [@a];
}
close REGIONS;

my $complete_area = 6201.77854968362;
my $total_area = 4*pi*((180/pi)**2);
my $cfrac = $complete_area/$total_area;
my $incompleteness = 0.08;
my $icorr = 1.0/(1-$incompleteness);
my $mean = 0.00033;
my $a_s = 2.91e-3;
my $b_s = 1.44e-4;
my $same = ($mean - $b_s)/$a_s;
my $vcorr_dr7 = 1.11340056672393;

print "#RA Dec Z SM SSFR ND Rmag GRcolor RIcolor Size[kpc] DR7PhotoObjID\n";
while (<>) {
    next unless (/^\d/);
    chomp;
    my @a = split(/,/);
    my ($ra, $dec, $z, $sm, $ssfr, $r, $gr, $ri, $size, $dr7) = @a[1,2,3,5,7,9,10,11,12,19];
    next unless ($z>0.01 and complete($sm, $z));
    next unless (check_regions($ra, $dec));
    $size *= 4.848136811097624809e-6; #To radians
    $size *= Universe::Distance::transverse_distance($z)*1e3;
    $gr -= $r;
    $ri = $r - $ri;
    $r -= 5*log(Universe::Distance::luminosity_distance($z)/1e-5)/log(10);
    my $vw = $icorr/($cfrac*volume($sm));
    $vw *= $vcorr_dr7;
    $vw *= $mean / ($a_s*$z+$b_s) if ($z < $same);
    $sm += $LK_TO_LC;
    print "$ra $dec $z $sm $ssfr $vw $r $gr $ri $size $dr7\n";
}





sub complete {
    my ($sm, $z) = @_;
    my $abs_lum = -0.2 - 1.9*$sm;
    my $ld = Universe::Distance::luminosity_distance($z);
    my $rel_lim = $abs_lum + log($ld/10e-6)/log(10)*5;
    return 0 if ($rel_lim > $pr_lim);
    return 1;
}

sub check_regions {
    my ($ra, $dec) = @_;
    for (@regions) {
        return 1 if ($ra > $_->[0] and $ra < $_->[2] and
                     $dec > $_->[1] and $dec < $_->[3]);
    }
    return 0;
}


sub volume {
    my ($sm) = @_;
    my $z1 = 0.0001;
    my $z2 = 1.0;
    my $z = 0.5*($z1+$z2);
    while ($z-$z1 > 0.00001) {
        my $c = complete($sm, $z);
        if ($c) { $z1 = $z; }
        else { $z2 = $z; }
        $z = 0.5*($z1+$z2);
    }
#    $z=0.05 if ($z>0.05);
    return Universe::Distance::comoving_volume($z)-Universe::Distance::comoving_volume(0.01);
}
