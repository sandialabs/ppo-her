for d in */ ; do
  cd "$d"
  pwd
  python process.py
  cd ..
done


