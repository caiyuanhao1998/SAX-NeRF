import * as THREE from 'three';
import {OrbitControls} from './vendor/three/addons/controls/OrbitControls.js';
import {GLTFLoader} from './vendor/three/addons/loaders/GLTFLoader.js';

export class CTMeshViewer {
  constructor(canvas, onCamera, onSlice) {
    this.canvas = canvas;
    this.onCamera = onCamera;
    this.onSlice = onSlice;
    this.meta = null;
    this.serial = 0;
    this.opacity = .38;
    this.showPlane = true;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#090e14');
    this.renderer = new THREE.WebGLRenderer({canvas, antialias: true, alpha: false});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.15;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.camera = new THREE.PerspectiveCamera(36, 1, .0001, 100);
    this.camera.position.set(1, .35, -.7);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = .13;
    this.controls.enablePan = false;
    this.controls.minPolarAngle = THREE.MathUtils.degToRad(5);
    this.controls.maxPolarAngle = THREE.MathUtils.degToRad(175);
    this.controls.addEventListener('change', () => this.onCamera?.(this.getView()));
    this.scene.add(new THREE.HemisphereLight(0xd6e7f5, 0x3a414b, 2.1));
    this.keyLight = new THREE.DirectionalLight(0xffffff, 2.4);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(1024, 1024);
    this.keyLight.shadow.bias = -.0002;
    this.scene.add(this.keyLight);
    const fill = new THREE.DirectionalLight(0xa9d9e9, 1.3);
    fill.position.set(-2, 1, -2);
    this.scene.add(fill);
    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShadowMaterial({color: 0x000000, opacity: .32}));
    this.floor.rotation.x = -Math.PI / 2;
    this.floor.receiveShadow = true;
    this.scene.add(this.floor);
    this.plane = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({
      color: 0x49b1cb, side: THREE.DoubleSide, transparent: true, opacity: .12,
      depthWrite: false, depthTest: false, toneMapped: false,
    }));
    this.plane.renderOrder = 4;
    this.scene.add(this.plane);
    this.outline = new THREE.LineLoop(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0x4fd5ea, depthTest: false}));
    this.outline.renderOrder = 5;
    this.scene.add(this.outline);
    this.rail = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0x4fd5ea, transparent: true, opacity: .45, depthTest: false}));
    this.rail.renderOrder = 5;
    this.scene.add(this.rail);
    this.handle = new THREE.Mesh(new THREE.SphereGeometry(1, 18, 12), new THREE.MeshBasicMaterial({color: 0x56daef, depthTest: false}));
    this.handle.renderOrder = 6;
    this.scene.add(this.handle);
    this.raycaster = new THREE.Raycaster();
    this.setPlaneVisible(false);
    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener('pointerdown', event => this.beginDrag(event), {capture: true});
    canvas.addEventListener('pointermove', event => this.drag(event), {capture: true});
    for (const type of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(type, event => this.endDrag(event), {capture: true});
    }
    this.animate();
  }

  resize() {
    const {width, height} = this.canvas.parentElement.getBoundingClientRect();
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(height, 1);
    this.camera.updateProjectionMatrix();
  }

  animate() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(() => this.animate());
  }

  disposeObject(object) {
    object.traverse(child => {
      child.geometry?.dispose();
      const materials = child.material ? (Array.isArray(child.material) ? child.material : [child.material]) : [];
      for (const material of materials) material.dispose();
    });
  }

  async load(meta) {
    const serial = ++this.serial;
    this.ready = false;
    this.meta = meta;
    this.rotation = new THREE.Matrix3().set(...meta.display_rotation.flat());
    this.half = meta.physical_size.map(n => n / 2);
    this.size = Math.max(...meta.physical_size);
    if (this.model) {
      this.scene.remove(this.model);
      this.disposeObject(this.model);
      this.model = null;
    }
    this.setPlaneVisible(false);
    const gltf = await new GLTFLoader().loadAsync(meta.mesh_url);
    if (serial !== this.serial) {
      this.disposeObject(gltf.scene);
      return false;
    }
    this.model = gltf.scene;
    const nativeGLB = new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 1);
    const display = new THREE.Matrix4().setFromMatrix3(this.rotation);
    this.model.applyMatrix4(display.multiply(nativeGLB.invert()));
    this.model.traverse(child => {
      if (child.isMesh) {
        child.castShadow = true;
        child.material = child.material.clone();
        child.material.side = THREE.DoubleSide;
        child.material.transparent = true;
        child.material.depthWrite = false;
        child.material.opacity = this.opacity;
      }
    });
    this.scene.add(this.model);
    this.floor.scale.setScalar(this.size * 4);
    const worldHalf = this.nativeToWorld(this.half).toArray().map(Math.abs);
    this.floor.position.set(0, -worldHalf[1] - this.size * .025, 0);
    this.keyLight.position.set(this.size * 2, this.size * 3, this.size * 2);
    Object.assign(this.keyLight.shadow.camera, {left: -this.size, right: this.size,
      top: this.size, bottom: -this.size, near: this.size * .01, far: this.size * 12});
    this.keyLight.shadow.camera.updateProjectionMatrix();
    this.handle.scale.setScalar(this.size * .018);
    this.reset();
    this.ready = true;
    return true;
  }

  reset() {
    if (!this.meta) return;
    const theta = THREE.MathUtils.degToRad(35), phi = THREE.MathUtils.degToRad(15);
    const eye = new THREE.Vector3(Math.cos(phi) * Math.cos(theta), Math.sin(phi), -Math.cos(phi) * Math.sin(theta));
    const forward = eye.clone().negate();
    const right = forward.clone().cross(new THREE.Vector3(0, 1, 0)).normalize();
    const up = right.clone().cross(forward).normalize();
    const half = this.nativeToWorld(this.half);
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2));
    let distance = 0;
    for (const x of [-half.x, half.x]) for (const y of [-half.y, half.y]) for (const z of [-half.z, half.z]) {
      const corner = new THREE.Vector3(x, y, z);
      distance = Math.max(distance, corner.dot(eye) + Math.abs(corner.dot(right)) / (tangent * this.camera.aspect * .82),
        corner.dot(eye) + Math.abs(corner.dot(up)) / (tangent * .82));
    }
    this.camera.position.copy(eye.multiplyScalar(distance));
    this.controls.target.set(0, 0, 0);
    this.controls.minDistance = distance * .35;
    this.controls.maxDistance = distance * 4;
    this.camera.near = this.size * .001;
    this.camera.far = distance * 20;
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.onCamera?.(this.getView());
  }

  getView() {
    const p = this.camera.position.clone().sub(this.controls.target);
    return {azimuth: (THREE.MathUtils.radToDeg(Math.atan2(-p.z, p.x)) + 360) % 360,
      elevation: THREE.MathUtils.radToDeg(Math.asin(p.y / p.length()))};
  }

  nativeToWorld(point) {
    return new THREE.Vector3(...point).applyMatrix3(this.rotation);
  }

  setOpacity(value) {
    this.opacity = Number(value);
    this.model?.traverse(child => {if (child.isMesh) child.material.opacity = this.opacity;});
  }

  setPlaneVisible(value) {
    this.showPlane = value;
    for (const object of [this.plane, this.outline, this.rail, this.handle]) object.visible = value && Boolean(this.meta);
  }

  updatePlane(axis, index) {
    if (!this.meta) return;
    this.axis = axis;
    this.index = index;
    const {u, v, u_sign: uSign, v_sign: vSign} = this.meta.slice_planes[axis];
    const position = this.meta.origin[axis] + index * this.meta.spacing[axis];
    const points = [];
    for (const [a, b] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
      const native = [0, 0, 0];
      native[axis] = position;
      native[u] = a * this.half[u] * uSign;
      native[v] = b * this.half[v] * vSign;
      points.push(this.nativeToWorld(native));
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(points.flatMap(p => p.toArray()), 3));
    geometry.setIndex([0, 1, 2, 0, 2, 3]);
    geometry.computeVertexNormals();
    this.plane.geometry.dispose();
    this.plane.geometry = geometry;
    this.outline.geometry.dispose();
    this.outline.geometry = new THREE.BufferGeometry().setFromPoints(points);
    const offset = [0, 0, 0];
    offset[u] = this.half[u] * 1.04 * uSign;
    offset[v] = this.half[v] * .94 * vSign;
    const a = [...offset], b = [...offset], handle = [...offset];
    a[axis] = this.meta.origin[axis];
    b[axis] = this.meta.origin[axis] + (this.meta.shape[axis] - 1) * this.meta.spacing[axis];
    handle[axis] = position;
    this.railStart = this.nativeToWorld(a);
    this.railEnd = this.nativeToWorld(b);
    this.rail.geometry.dispose();
    this.rail.geometry = new THREE.BufferGeometry().setFromPoints([this.railStart, this.railEnd]);
    this.handle.position.copy(this.nativeToWorld(handle));
    this.setPlaneVisible(this.showPlane);
  }

  pointer(event) {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -(event.clientY - rect.top) / rect.height * 2 + 1);
  }

  beginDrag(event) {
    if (!this.ready || !this.showPlane) return;
    this.raycaster.setFromCamera(this.pointer(event), this.camera);
    if (!this.raycaster.intersectObject(this.handle).length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    this.dragging = {id: event.pointerId, startY: event.clientY, startIndex: this.index};
    this.controls.enabled = false;
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.style.cursor = 'ns-resize';
  }

  drag(event) {
    if (!this.ready || !this.showPlane) return;
    if (!this.dragging) {
      this.raycaster.setFromCamera(this.pointer(event), this.camera);
      this.canvas.style.cursor = this.raycaster.intersectObject(this.handle).length ? 'ns-resize' : 'grab';
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
    const rect = this.canvas.getBoundingClientRect();
    const a = this.railStart.clone().project(this.camera), b = this.railEnd.clone().project(this.camera);
    const start = new THREE.Vector2((a.x + 1) * rect.width / 2, (1 - a.y) * rect.height / 2);
    const end = new THREE.Vector2((b.x + 1) * rect.width / 2, (1 - b.y) * rect.height / 2);
    const direction = end.sub(start), pointer = new THREE.Vector2(event.clientX - rect.left, event.clientY - rect.top).sub(start);
    const total = this.meta.shape[this.axis] - 1;
    const index = direction.lengthSq() > 160 ? pointer.dot(direction) / direction.lengthSq() * total
      : this.dragging.startIndex - (event.clientY - this.dragging.startY) / (rect.height * .7) * total;
    this.onSlice?.(Math.max(0, Math.min(total, Math.round(index))));
  }

  endDrag(event) {
    if (!this.dragging) return;
    this.dragging = null;
    this.controls.enabled = true;
    this.canvas.style.cursor = 'grab';
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
  }

  handleScreenPosition() {
    const rect = this.canvas.getBoundingClientRect();
    const point = this.handle.position.clone().project(this.camera);
    return {x: rect.left + (point.x + 1) * rect.width / 2, y: rect.top + (1 - point.y) * rect.height / 2};
  }
}
