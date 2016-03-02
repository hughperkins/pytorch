if false; then {
  git clone https://github.com/torch/distro.git ~/torch
  cd ~/torch
  'for pkg in cudnn cunn cunnx cutorch qttorch trepl graph optim sdl2 threads submodule graphicsmagick audio fftw3 signal nnx qtlua gnuplot dok iTorch argcheck image xlua; do { sed -i -e "s/\(.*$pkg.*\)/echo skipping $pkg # \1/" install.sh; } done'
  'awk ''NR==2{print "set -x"}1'' install.sh > ~install.sh'
  mv ~install.sh install.sh
  chmod +x install.sh
  cat install.sh
  for pkg in exe/luajit-rocks extra/nn pkg/cwrap pkg/paths pkg/sundown pkg/sys pkg/torch pkg/paths extra/lua-cjson extra/luaffifb extra/luafilesystem extra/penlight; do { git submodule update --quiet --init $pkg; } done
  sed -i -e 's/$(MAKE)/$(MAKE) -j 4/' pkg/torch/rocks/torch-scm-1.rockspec
  ./install.sh -b >/dev/null
} else {
  mkdir -p ~/torch
  cd ~/torch
  wget https://s3.amazonaws.com/hughperkinstravis/hughperkins/distro/3/3.1/torch-install.tar.bz2
  tar -xf torch-install.tar.bz2
} fi

sed -i -e 's/^export LD_LIBRARY_PATH/# export LD_LIBRARY_PATH/' ~/torch/install/bin/torch-activate
sed -i -e 's/^export DYLD_LIBRARY_PATH/# export LD_LIBRARY_PATH/' ~/torch/install/bin/torch-activate
source ~/torch/install/bin/torch-activate
luajit -l torch -e 'print(torch.Tensor(3,2):uniform())'

