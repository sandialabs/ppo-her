for d in */ ; do
  cd "$d"
  pwd
  python calc_stats.py
  cd ..
done
