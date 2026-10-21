#include <GLUT/glut.h>								//GLUT library
#include "UnstructuredGrid3D.h"							//the 3D unstructured mesh class
#include "MeshRenderer.h"							//a simple renderer for 3D unstructured grids, demonstrating flat/smooth shading
#include "PlyReader.h"
#include "zpr.h"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <unistd.h>


using namespace std;



MeshRenderer		renderer;						//Renderer to draw the meshes
UnstructuredGrid3D*	mesh			=  0;				//Original mesh
UnstructuredGrid3D*	hull			=  0;				//Convex hull geometry
UnstructuredGrid3D*	show_what		=  0;				//Either mesh or hull, tells what we'll view

int			drawing_style   	=  0;				//Combined drawing style of mesh and hull
										//0: draw just mesh
										//1: draw transparent hull atop mesh
										//2: draw transparenrt hull + its wireframe atop mesh
										//3: draw shaded hull
										//4: draw shaded hull + its wireframe											

static void mouseclick(int button, int state, int x, int y)
{
	int keys = glutGetModifiers();						//The mouse events are interpreted as follows:	
	if (keys & GLUT_ACTIVE_CTRL)						// 
	button = GLUT_MIDDLE_BUTTON;						// -left button + move:                         rotate viewpoint
	if (keys & GLUT_ACTIVE_SHIFT)						// -middle button (or left button+Control key): zoom in / out
	button = GLUT_RIGHT_BUTTON;							// -right buttom (or left button+Shift key):    translate viewpoint

	zprMouse(button,state,x,y);							//Use the ZPR library to manipulate the viewpoint. The library sets the modelview  
														//OpenGL matrix based on the mouse events, thereby changing the current viewpoint.
}



static void computeHull(UnstructuredGrid3D* input, UnstructuredGrid3D*& output)	//Compute convex hull of input, save it into output
{
	int NP=input->numPoints();
	float p[3];

	FILE* fo = fopen("temp.inp.txt","w");					//Make a temporary file to communicate 3D points to qhull
	fprintf(fo,"3\n");
	fprintf(fo,"%d\n",NP);							//#points in our input mesh
	
	for(int i=0;i<NP;++i)							//save all points to file
	{
	   input->getPoint(i,p);
	   fprintf(fo,"%f %f %f\n",p[0],p[1],p[2]);
	}	
	fclose(fo);								//file ready
	
	char cmdline[255];							//call qhull to compute hull (triangulated) and save it to another file
	sprintf(cmdline, "./qhull/bin/qhull TI temp.inp.txt QJ o TO temp.out.txt");	
	int ret = system(cmdline);	
	if (ret) return;							//if qhull failed executing, don't continue
	
	fo = fopen("temp.out.txt","r");						//read qhull's result
	
	int NP2,NF2;
	fscanf(fo,"%*d %d %d %*d", &NP2,&NF2);					//get the #points and #triangles in hull
	
	delete output;								//delete old output since we're making a new one
	output = new UnstructuredGrid3D(NP2,NF2);				//make new unstructured grid, fill it next with hull 
	
	for(int i=0;i<NP2;++i)							//add points to hull (these are actually the points from 'input'
	{
		fscanf(fo,"%f %f %f",&p[0],&p[1],&p[2]);
		output->setPoint(i,p);
	}
	
	for(int i=0;i<NF2;++i)							//add triangles to hull
	{
		int c[3];
		fscanf(fo,"%*d %d %d %d",&c[0],&c[1],&c[2]);
		output->setCell(i,c);		
	}
	
	fclose(fo);								//done with the temp file
	unlink("temp.inp.txt");							//delete the temporary files since we're done with them
	unlink("temp.out.txt");
	
	output->computeFaceNormals();						//compute face and vertex normals for the mesh. This allows us to shade the mesh next.
	output->computeVertexNormals();								
}



static void mousemotion(int x, int y)
{
	zprMotion(x,y);
	glutPostRedisplay();
}


static void viewing(int width, int height)					//GLUT window-reshape callback
{
	glutReshapeWindow(width,height);  
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60,float(width)/height,-100,100);
	glViewport(0,0,width,height);
}


static void keyboard(unsigned char ch,int,int)				//GLUT keyboard callback
{
	switch(ch)
	{
	case 'h':
		{
		   computeHull(mesh,hull);
		   drawing_style = 1;
		   break;
		 }						
	case ' ':												//Toggle through the various drawing styles of the mesh renderer
		{
		  drawing_style = (++drawing_style) % 5;		
		  break;
		}
	case 'q':
	case 27:
		exit(0);	
	}	
	
	glutPostRedisplay();	
}		


static void display()										//GLUT display callback
{
   glClearColor(1,1,1,1);
   glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

   renderer.setDrawingStyle(MeshRenderer::DRAW_C1_CELLS);					//Draw mesh (always with smooth shading
   renderer.draw(*mesh);   

   if (drawing_style>0)
   {
      if (drawing_style>2)
      {
         glEnable(GL_BLEND);										
         glBlendFunc(GL_CONSTANT_ALPHA,GL_ONE_MINUS_CONSTANT_ALPHA);
         glBlendColor(0.6,0.6,0.6,0.6);
      }

      if (drawing_style==1 || drawing_style==3)
           renderer.setDrawingStyle(MeshRenderer::DRAW_C1_CELLS);
      if (drawing_style==2 || drawing_style==4)
           renderer.setDrawingStyle(MeshRenderer::DRAW_GRID);      

      renderer.draw(*hull);
      glDisable(GL_BLEND);
   }

   glutSwapBuffers();
}



int main(int argc,char* argv[])								//Main program
{
	cout<<"Convex hull computation"<<endl;
	cout<<"In the graphics window, press"<<endl;
	cout<<"   space:  Toggle drawing styles for hull and mesh"<<endl;
	cout<<"   h:      Compute and display the hull atop the mesh"<<endl;   
		
	const char* filename = (argc>1)? argv[1]:"DATA/bunny.ply";
	
	glutInit(&argc, argv);									//1. Initialize the GLUT toolkit
	glutInitDisplayMode(GLUT_RGB|GLUT_DOUBLE|GLUT_DEPTH);	//2. Ask GLUT to create next windows with a RGB framebuffer and a Z-buffer too
	glutInitWindowSize(500,500);							//3. Tell GLUT how large are the windows we want to create next
	glutCreateWindow("Convex hull computation");		


	PlyReader rdr;											//6.  Read a 3D mesh stored in a file in the PLY format
        mesh = rdr.read(filename);
        mesh->normalize();										//7.  Normalize the mesh in the [-1,1] cube. This makes setting the OpenGL projection easier.
        mesh->computeFaceNormals();								//8.  Compute face and vertex normals for the mesh. This allows us to shade the mesh next.
        mesh->computeVertexNormals();
	mesh->computePointTriangles();

	hull = new UnstructuredGrid3D(0,0);


	zprInit(0,0,0);


	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouseclick);
	glutMotionFunc(mousemotion);
	glutReshapeFunc(viewing);
	glutDisplayFunc(display);
	
	glutMainLoop();

	return 0;
}




