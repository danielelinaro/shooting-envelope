
TAGS = ${HOME}/local/bin/ctags
RM = rm -f

tags: FORCE
	$(RM) tags
	$(TAGS) -f tags -w *.py */*.py ./*/*.py

FORCE:	
	
