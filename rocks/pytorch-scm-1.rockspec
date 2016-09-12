package = "pytorch"
version = "scm-1"

source = {
   url = "git://github.com/hughperkins/pytorch.git",
}

description = {
   summary = "Python wrappers for torch/lua",
   detailed = [[
   ]],
   homepage = "https://github.com/hughperkins/pytorch",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0"
}

build = {
   type = "command",
   build_command = [[
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"  -DLUA_INCDIR="$(LUA_INCDIR)" -DLUA_LIBDIR="$(LUA_LIBDIR)" && $(MAKE)
]],
   install_command = "cd build && $(MAKE) install"
}
