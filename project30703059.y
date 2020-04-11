%{
#include<stdio.h>
#include<stdlib.h>
#include<fcntl.h>
#include<string.h>
%}
%token OK
%right UMINUS
%%
ST : OK {printf("\nSTRINGS ARE THE SAME\n SIMILAR CONTENT!!!!!"); return 0;}
   ;
%%
#include"lex.yy.c"
int main(void)
{
printf("\nWELCOME TO THE SENTENCE SIMILARITY CHECKER\n");
printf("\n\nPLEASE NOTE: ENTER THE STRINGS IN THE FOLLOWING MANNER:");
printf("\n\nEXAMPLE:The quick brown fox .\n\nEND1(###to view entered 1st string and its length###)\nThe quick brown dog .\n\nEND2(###to view entered 2nd string and its length###)");
printf("\n\nENTER THE SIZE OF THE FIRST STRING");
scanf("%d",&len1);
printf("\n\nENTER THE SIZE OF THE SECOND STRING");
scanf("%d",&len2);
yyparse();
}
int yyerror(char * s)
{
printf("\nSTRINGS ARE DIFFERENT \nDISSIMILAR CONTENT!!!!");
return 0;
}





