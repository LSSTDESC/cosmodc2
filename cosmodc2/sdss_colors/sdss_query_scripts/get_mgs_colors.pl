#!/usr/bin/perl -w

$last_id = "0";
$lines = 0;
$maxlines = 300000;
do {
    submit_request($last_id);
    ($lines) = (split(" ", scalar `wc out.csv`));
    my $last_line = `tail -n1 out.csv`;
    if (!$last_id) {
	rename "out.csv", "dr10_mgs_colors.txt";
    } else {
	system("tail -n $maxlines out.csv >> dr10_mgs_colors.txt");
	unlink "out.csv";
    }
    ($last_id) = split(",", $last_line);
} while ($lines == ($maxlines+2));


sub submit_request {
    my $last_id = shift;
$data = <<"EOL"
SELECT TOP $maxlines
sp.specObjID,sp.ra,sp.dec,sp.z,sp.zWarning,g.lgm_tot_p50,g.sfr_tot_p50,g.specsfr_tot_p50,g.bptclass,sp.petroMag_r,sp.petroMag_g,sp.petroMag_i,p.petroR50_r,sp.cModelMag_r,sp.cModelMag_g,sp.dered_r,sp.dered_g,sp.modelMag_r,sp.modelMag_g,d.dr7objid
FROM SpecPhotoAll AS sp
   JOIN galSpecExtra AS g ON sp.specObjID = g.specObjID
   JOIN PhotoObjAll AS p ON sp.objID = p.objID
   JOIN PhotoObjDR7 AS d ON d.dr8objid = p.objID
--   JOIN galSpecIndx AS gs ON sp.specObjID = gs.specObjID
WHERE 
   sp.sourceType = 'GALAXY' AND
   sp.petroMag_r < 17.77 AND
   sp.sciencePrimary > 0 AND
   sp.specObjID > $last_id
ORDER BY sp.specObjID
EOL
    ;

    $data =~ s/([^a-zA-Z0-9])/"\%".uc(unpack("H*", $1))/eg;

    system("curl -o out.csv 'http://skyserver.sdss.org/dr14/en/tools/search/x_results.aspx?submit=submit&format=csv&searchtool=SQL&TaskName=Skyserver.Search.SQL&NoSyntax=syntax&cmd=$data'"); 
}

