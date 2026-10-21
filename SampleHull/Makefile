
.SUFFIXES:	.o .c .cpp .cc .C .a


RM = -rm


.cpp.o:	
	$(C++) $(CCFLAGS) $*.cpp


#############################################################

C++        = g++
CCFLAGS    = -c -Iinclude 


VPATH   = Obj

OBJECTS1 = UnstructuredGrid3D.o UnstructuredGrid.o Grid.o MeshRenderer.o PlyReader.o plyfile.o ScalarAttributes.o zpr.o sampleHull.o

QHULL = qhull/bin/qhull

default: sampleHull

$(QHULL):
	cd qhull && $(MAKE) bin/qhull

sampleHull: $(OBJECTS1) $(QHULL)
	-mv $(OBJECTS1) Obj
	cd Obj ; $(C++) -o ../sampleHull $(OBJECTS1) -framework OpenGL -framework GLUT; cd .

clean:
	$(RM) $(VPATH)/*.o sampleHull
	cd qhull && $(MAKE) cleanall
 

