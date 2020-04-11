word [a-zA-Z|.|!|#|@|^|%|$|_|-|=|+|(|)|*|&]
%{
#include<string.h>
#include<fcntl.h>
#include<malloc.h>
#include <stdio.h>
#include <stdlib.h>
int len1,len2;
int res1,res2,res3;
struct words
{
char wo[10];
};
struct words1
{
char wo1[10];
};
struct words2
{
char wo2[10];
};
struct words3
{
char wo3[10];
};
struct words4
{
char wo4[10];
};
struct words5
{
char wo5[10];
};

struct node1
{
    struct node *prev;
    char w[10];
    struct node *next;
}*h1,*temp,*temp1,*temp2,*temp10;
 


void insert1(char *);
void traverse1();
int count1=0;
 
struct node2
{
    struct node1 *prev;
    char w[10];
    struct node1 *next;
}*h2,*temp5,*temp6,*temp7,*temp11;

void insert2(char *);
void traverse2();
int count2=0;


int compare(); 
void create1(char *input1)
{
 
    temp =(struct node1 *)malloc(1*sizeof(struct node1));
    temp->prev = NULL;
    temp->next = NULL;
    strcpy(temp->w,input1);
    count1++;
}
void create2(char *input2)
{
    

    temp5 =(struct node2 *)malloc(1*sizeof(struct node2));
    temp5->prev = NULL;
    temp5->next = NULL;
    strcpy(temp5->w,input2);
    count2++;
}

void insert1(char *input1)
{   
    if (h1 == NULL)
    {
        create1(input1);
        h1 = temp;
        temp1 = h1;
    }
    else
    {  if((count1==len1)&&(count2<=len2))
       {insert2(input1);}
       else{ 
        if(count1<=len1){
       create1(input1);
        temp1->next = temp;
        temp->prev = temp1;
        temp1 = temp;}
}
    
}
}
void insert2(char *input2)
{
    if (h2 == NULL)
    {
        create2(input2);
        h2 = temp5;
        temp6 = h2;
    }
    else
    {
        create2(input2);
        temp6->next = temp5;
        temp5->prev = temp6;
        temp6 = temp5;
    }
}




void traverse1()
{
    temp2 = h1;


     if (temp2 == NULL)
    { printf("\nNO SENTENCE FOUND!");
        return;
    }
    printf("\nSENTENCE 1");
    printf("\n");
    while (temp2->next != NULL)
    {
        printf(" %s", temp2->w);
        temp2 = temp2->next;
    }
    printf(" %s ", temp2->w);

}

void traverse2()
{
    temp7 = h2;
    
    if (temp7 == NULL)
    {printf("\nNO SENTENCE FOUND!");

        return;
    }
  printf("\nSENTENCE 2");
    printf("\n");

    while (temp7->next != NULL)
    {
        printf(" %s", temp7->w);
        temp7 = temp7->next;
    }
    printf(" %s ", temp7->w);
}
int comparecurrent()
{
int flag=0;
if((len1>len2)||(len2>len1))
flag=1;
else
{
temp10=h1;
temp11=h2;
while((temp10->next!=NULL)||(temp11->next!=NULL))
{
if(strcmp(temp10->w,temp11->w)==0)
{
flag=0;
temp10=temp10->next;
temp11=temp11->next;
}
else
{flag=1;
temp10=temp10->next;
temp11=temp11->next;
}
}
}
return flag;
}
int compareprevious()
{
int flag=0;
if((len1>len2)||(len2>len1))
flag=1;
else
{
struct node *temps1,*temps2,*temps3,*temps4;
temps1=h1;
temps2=h2;
temps1=temps1->next;
temps2=temps2->next;
while((temps1->next!=NULL)||(temps2->next!=NULL))
{temps3=temps1->prev;
temps4=temps2->prev;
if(strcmp(temps3->w,temps4->w)==0)
{
flag=0;
temp10=temp10->next;
temp11=temp11->next;
}
else
{flag=1;
temp10=temp10->next;
temp11=temp11->next;
}
}
}
return flag;
}

int comparenext()
{
int flag=0;
if((len1>len2)||(len2>len1))
flag=1;
else
{
struct node *temps1,*temps2,*temps3,*temps4;
temps1=h1;
temps2=h2;
temps1=temps1->next;
temps2=temps2->next;
while((temps1->next!=NULL)||(temps2->next!=NULL))
{temps3=temps1->next;
temps4=temps2->next;
if(strcmp(temps3->w,temps4->w)==0)
{
flag=0;
temp10=temp10->next;
temp11=temp11->next;
}
else
{flag=1;
temp10=temp10->next;
temp11=temp11->next;
}
}
}
return flag;
}


void wordset()
{
struct words W[count1];
struct words1 W1[count2];
struct words2 W2[count1];
struct words3  W3[count2];
struct words4  W4[count1];
struct words5  W5[count1+count2];
int i=0,j=0,l=0,count4;
int co1=0,co2=0,co3=0;
temp10=h1;
temp11=h2;
printf("\n");
for(i=0;i<count1;i++)
{strcpy(W[i].wo,temp10->w);
temp10=temp10->next;
}
printf("\n");
for(i=0;i<count2;i++)
{strcpy(W1[i].wo1,temp11->w);
temp11=temp11->next;
}
for(i=0;i<count1;i++)
{
for(j=0;j<co1;j++)
{
if(strcasecmp(W[i].wo,W2[j].wo2)==0)
break;
}
if(j==co1)
{strcpy(W2[co1].wo2,W[i].wo);
co1++;
}
}
printf("\n\nTHE SUBSET OF WORDS IN SENTENCE 1");
printf("\n");
printf("{");
for(i=0;i<co1;i++){
printf("%s",W2[i].wo2);
if(i!=(co1-1))
printf(",");

}
printf("}");
for(i=0;i<count2;i++)
{
for(j=0;j<co2;j++)
{
if(strcasecmp(W1[i].wo1,W3[j].wo3)==0)
break;
}
if(j==co2)
{strcpy(W3[co2].wo3,W1[i].wo1);
co2++;
}
}
printf("\n\nTHE SUBSET OF WORDS IN SENTENCE 2");
printf("\n");
printf("{");
for(i=0;i<co2;i++){
printf("%s",W3[i].wo3);
if(i!=(co2-1))
printf(",");

}
printf("}");
count4=count1+count2;
for(i=0;i<count1;i++)
{strcpy(W4[i].wo4,W[i].wo);
}
for(i=count1;i<count4;i++)
{strcpy(W4[i].wo4,W1[l].wo1);
l++;
}
for(i=0;i<count4;i++)
{
for(j=0;j<co3;j++)
{
if(strcasecmp(W4[i].wo4,W5[j].wo5)==0)
break;
}
if(j==co3)
{strcpy(W5[co3].wo5,W4[i].wo4);
co3++;
}
}
printf("\n\nTHE TOTAL SET OF WORDS IN BOTH SENTENCES");
printf("\n");
printf("{");
for(i=0;i<co3;i++){

printf("%s",W5[i].wo5);
if(i!=(co3-1))
printf(",");

}
printf("}");
}

%}
%%
{word}+             {insert1(yytext);
                     }
                    
              

                   
[\n]"END1"                    { printf("\n");
                      traverse1();
                    printf("\nTHE LENGTH OF THE FIRST SENTENCE IS\n");
                    printf("\n%d\t\t",count1);
                      }
             
[\n]"END2"                    { printf("\n"); 
                      traverse2();
                    printf("\nTHE LENGTH OF THE SECOND  SENTENCE IS\n");
                    printf("\n%d\t\t",count2);
                                       }


[\n]"COMPARE"                  {  wordset();  
                                 res1=comparecurrent();
                                 res2=compareprevious();
                                 res3=comparenext();

                                 if((res1==0)&&(res2==0)&&(res3==0))
                                 { return OK;}
                               }


%%


             

