#!/usr/bin/perl

foreach my $line ( <STDIN> ) {
	chomp( $line );
	if($line !~ "^import"){
		$line =~ s/\[[^\]]+\]//g; 
		print "$line\n";
	}
}
