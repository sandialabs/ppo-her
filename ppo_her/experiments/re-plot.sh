for d in */ ; do
  cd "$d"
  pwd
  python plot.py
  cd ..
done
